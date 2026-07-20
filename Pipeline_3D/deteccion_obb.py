"""
Sprint E2E — Detección 3D: YOLO -> recorte -> OBB (Módulo P1)
-------------------------------------------------------------
Cierra el pipeline del equipo con el modelo entrenado por P3 (87% mAP):

  por cada frame del .svo:
    1. YOLO (best.pt) detecta objetos en la imagen RGB -> cajas 2D + clase + conf
    2. cada caja 2D recorta la nube de puntos ORGANIZADA del frame
       (mismo contrato de datos que recorte_mascara.py, pero la caja ya no es mock)
    3. limpieza (voxel + outliers + RANSAC plano + DBSCAN) y OBB anclada al plano
       (misma lógica validada de limpieza_obb.py)

  SALIDA: detecciones_3d.csv con clase, confianza, centro (m), dimensiones (m) y
          orientación por objeto; opcionalmente los .ply de cada objeto.

Las dimensiones son una ESTIMACIÓN derivada del mapeo (~10-15% error),
NO una medición certificada.

Requisitos: además del venv zed3d (pyzed, open3d, opencv, numpy):
    pip install ultralytics

Uso:
    # un solo frame, con visor 3D (buen primer test):
    python deteccion_obb.py "C:\\ruta\\tu.svo" --modelo best.pt --frame 100 --show

    # todo el recorrido, procesando 1 de cada 15 frames:
    python deteccion_obb.py "C:\\ruta\\tu.svo" --modelo best.pt --cada 15

    # igual, pero con las cajas en coordenadas del MAPA (positional tracking):
    python deteccion_obb.py "C:\\ruta\\tu.svo" --modelo best.pt --cada 15 --mundo
"""

import os
import sys
import csv
import argparse
import numpy as np
import cv2
import pyzed.sl as sl
import open3d as o3d
from ultralytics import YOLO

# Reutilizamos la OBB anclada al plano ya validada (mismo folder del proyecto)
from limpieza_obb import obb_alineada_al_plano

# Carpeta de salida escribible (la del proyecto en Cowork es de solo lectura)
DIR = os.path.join(os.path.expanduser("~"), "Proyecto_ZED_P1")
os.makedirs(DIR, exist_ok=True)

# Colores vivos para distinguir cada objeto detectado en el visor 3D
PALETA = [(0.90, 0.10, 0.10), (0.10, 0.60, 0.95), (0.10, 0.80, 0.30),
          (0.95, 0.75, 0.10), (0.80, 0.20, 0.80), (0.95, 0.45, 0.10)]
JPG_DIR = os.path.join(DIR, "etiquetado")   # frames anotados (--guardar-jpg)


def quat_a_R(q):
    """Cuaternión (x,y,z,w) -> matriz de rotación 3x3 (igual que mapeo_manual)."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def solape_relativo(a, b):
    """Solape entre dos cajas 2D: área de intersección / área de la caja MENOR.
    (Mejor que IoU cuando una caja está casi contenida en la otra.)"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter == 0:
        return 0.0
    amin = min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1))
    return inter / amin


def suprimir_solapadas(dets, umbral, umbral_cruzado=None):
    """MACRO primero: de cada grupo de cajas 2D sobrepuestas conserva SOLO la de
    mayor confianza; las demás se posponen (la oclusión fina se aborda después).
    umbral = solape mínimo entre cajas de la MISMA clase (duplicados de YOLO).
    umbral_cruzado = solape mínimo entre clases DISTINTAS (silla frente a mesa):
    más alto = más tolerante, ambas se procesan aunque se toquen. >=1 desactiva."""
    if umbral_cruzado is None:
        umbral_cruzado = umbral
    if (umbral >= 1.0 and umbral_cruzado >= 1.0) or len(dets) <= 1:
        return dets, []
    kept, descartadas = [], []
    for d in sorted(dets, key=lambda d: -d["conf"]):
        if any(solape_relativo(d["caja"], k["caja"]) >
               (umbral if k["clase"] == d["clase"] else umbral_cruzado)
               for k in kept):
            descartadas.append(d)
        else:
            kept.append(d)
    return kept, descartadas


def filtrar_banda_central(xyz, col, vv, uu, Hc, Wc, banda, frac=0.2, min_central=30):
    """Anti-oclusión: el objeto detectado ocupa el CENTRO de su caja 2D, así que
    estimamos su distancia con la mediana de los puntos centrales y descartamos
    lo que esté mucho más cerca (un oclusor delante) o más lejos (el fondo).
    banda = medio-ancho en metros; banda <= 0 desactiva el filtro."""
    if banda <= 0:
        return xyz, col
    dist = np.linalg.norm(xyz, axis=1)
    central = (np.abs(vv - Hc / 2.0) < frac * Hc) & (np.abs(uu - Wc / 2.0) < frac * Wc)
    if central.sum() < min_central:      # centro sin profundidad (p.ej. vidrio)
        return xyz, col
    med = np.median(dist[central])
    keep = np.abs(dist - med) < banda
    return xyz[keep], col[keep]


def limpiar_y_obb(obj_xyz, obj_rgb, args, normal_fallback=None):
    """Limpieza C-D-E de limpieza_obb.py sobre el recorte de UNA detección.
    Devuelve (objeto_pcd, obb) o (None, None) si no queda nada útil."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj_xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(obj_rgb.astype(np.float64))

    # C. voxel + outlier removal
    pcd = pcd.voxel_down_sample(voxel_size=args.voxel)
    if len(pcd.points) > 20:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    if len(pcd.points) < args.min_obj_pts:
        return None, None

    # D. quitar plano de soporte (si es una porción relevante del recorte)
    plane_normal = None
    if len(pcd.points) > 100:
        plano, inliers = pcd.segment_plane(distance_threshold=args.plano,
                                           ransac_n=3, num_iterations=1000)
        plane_normal = np.array(plano[:3])
        if 0.15 < len(inliers) / len(pcd.points) < 0.9:
            pcd = pcd.select_by_index(inliers, invert=True)

    # E. DBSCAN -> cluster más grande (el objeto)
    if len(pcd.points) < args.min_obj_pts:
        return None, None
    labels = np.array(pcd.cluster_dbscan(eps=args.eps, min_points=args.min_pts))
    if labels.max() >= 0:
        cuenta = np.bincount(labels[labels >= 0])
        pcd = pcd.select_by_index(np.where(labels == int(np.argmax(cuenta)))[0])
    if len(pcd.points) < args.min_obj_pts:
        return None, None

    # OBB anclada al plano (o al 'arriba' del mundo si no hubo plano en el recorte)
    normal = plane_normal if plane_normal is not None else normal_fallback
    if normal is not None:
        obb = obb_alineada_al_plano(pcd, normal)
    else:
        try:
            obb = pcd.get_minimal_oriented_bounding_box(robust=True)
        except RuntimeError:
            return None, None
    return pcd, obb


def main():
    ap = argparse.ArgumentParser(description="YOLO -> recorte -> OBB por frame")
    ap.add_argument("svo")
    ap.add_argument("--modelo", default=os.path.join(DIR, "best.pt"),
                    help="Ruta a best.pt entrenado por P3")
    ap.add_argument("--frame", type=int, default=-1,
                    help="Procesar SOLO este frame (modo prueba). -1 = recorrer el svo")
    ap.add_argument("--cada", type=int, default=15,
                    help="En modo recorrido: correr YOLO 1 de cada N frames")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = todos")
    ap.add_argument("--conf", type=float, default=0.4, help="Confianza mínima YOLO")
    ap.add_argument("--clases", nargs="*", default=None,
                    help="Filtrar por nombre de clase (def. todas)")
    ap.add_argument("--max-dist", type=float, default=6.0,
                    help="Ignorar detecciones más lejas que esto (m)")
    ap.add_argument("--mundo", action="store_true",
                    help="Positional tracking: cajas en coords. del MAPA (como mapa_manual.ply)")
    ap.add_argument("--origen-camara", action="store_true",
                    help="Comportamiento viejo: origen del mundo donde arrancó la "
                         "cámara (por defecto ahora el origen es el PISO, z=0). "
                         "Usar el MISMO modo que en mapeo_manual.py.")
    ap.add_argument("--depth", choices=["neural_light", "neural", "neural_plus"],
                    default="neural",
                    help="Modo de profundidad ZED (neural_light = más rápido en la GTX 1650)")
    ap.add_argument("--zed-conf", type=int, default=50,
                    help="confidence_threshold ZED (1-100, MENOR = más estricto). Quita "
                         "profundidad dudosa de brillos/luz. Mismo valor que en mapeo_manual.")
    ap.add_argument("--zed-texconf", type=int, default=90,
                    help="texture_confidence_threshold ZED (1-100, menor = más estricto). "
                         "Quita zonas saturadas/sin textura (ventanas, brillos). "
                         "Si sigue entrando ruido de luz, baja a 80.")
    # Parámetros de limpieza (mismos defaults de limpieza_obb.py)
    ap.add_argument("--voxel", type=float, default=0.005)
    ap.add_argument("--plano", type=float, default=0.015)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--min-pts", type=int, default=20)
    ap.add_argument("--min-obj-pts", type=int, default=60,
                    help="Puntos mínimos para aceptar un objeto 3D")
    ap.add_argument("--solape", type=float, default=0.05,
                    help="Solape máximo entre cajas 2D de la MISMA clase antes de "
                         "quedarse solo con la de mayor confianza (duplicados de "
                         "YOLO sobre el mismo objeto). >=1 desactiva.")
    ap.add_argument("--solape-cruzado", type=float, default=0.60,
                    help="Ídem pero entre clases DISTINTAS (silla frente a mesa). "
                         "0.60 = solo se descarta si una caja está mayormente "
                         "contenida en la otra; tocarse ya no elimina detecciones. "
                         "(Antes se usaba el mismo umbral agresivo de --solape.)")
    ap.add_argument("--banda", type=float, default=0.0,
                    help="(experimental, apagado por defecto) Anti-oclusión fina: "
                         "medio-ancho (m) de la banda de profundidad alrededor del "
                         "centro de la caja 2D. 0 = desactivado.")
    ap.add_argument("--show", action="store_true",
                    help="Visor: RGB con cajas 2D + escena 3D con OBBs (pausa por frame)")
    ap.add_argument("--guardar-ply", action="store_true",
                    help="Guardar la nube limpia de cada objeto (objeto_fXXXX_N.ply)")
    ap.add_argument("--guardar-jpg", action="store_true",
                    help="Guardar cada frame procesado como imagen anotada (cajas 2D + "
                         "clase + confianza) en Proyecto_ZED_P1\\etiquetado\\ — para "
                         "revisar QUÉ está etiquetando el YOLO")
    ap.add_argument("--csv", default=os.path.join(DIR, "detecciones_3d.csv"))
    args = ap.parse_args()

    if args.mundo and args.frame >= 0:
        print("[WARN] --mundo necesita recorrer el svo desde el inicio (el seek rompe "
              "el tracking); ignoro --frame.")
        args.frame = -1

    if not os.path.isfile(args.modelo):
        print(f"[ERROR] No encuentro el modelo: {args.modelo}\n"
              f"        Descarga best.pt del Drive de P3 y pásalo con --modelo.")
        sys.exit(1)

    print(f"[OK] Cargando YOLO: {args.modelo}")
    model = YOLO(args.modelo)
    nombres = model.names
    print(f"[OK] Clases del modelo ({len(nombres)}): {list(nombres.values())}")

    # --- Abrir el SVO (misma config validada: NEURAL, metros, Z arriba) ---
    DEPTH = {"neural_light": sl.DEPTH_MODE.NEURAL_LIGHT,
             "neural": sl.DEPTH_MODE.NEURAL,
             "neural_plus": sl.DEPTH_MODE.NEURAL_PLUS}[args.depth]

    init = sl.InitParameters()
    init.set_from_svo_file(args.svo)
    init.svo_real_time_mode = False
    init.depth_mode = DEPTH
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
    init.depth_maximum_distance = args.max_dist + 2.0

    zed = sl.Camera()
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("[ERROR] No se pudo abrir el .svo"); sys.exit(1)
    total = zed.get_svo_number_of_frames()
    print(f"[OK] SVO abierto. Frames: {total}", flush=True)

    if args.mundo:
        track_par = sl.PositionalTrackingParameters()
        if not args.origen_camara:
            # PISO como origen: z=0 en el suelo, igual que mapeo_manual.py.
            # Así el mapa y las cajas comparten marco aunque la cámara haya
            # arrancado a cualquier altura, y los flags de consolidar valen.
            track_par.set_floor_as_origin = True
            print("[OK] Origen del mundo = PISO (set_floor_as_origin). "
                  "El mapa debe generarse en el MISMO modo.", flush=True)
        if zed.enable_positional_tracking(track_par) != sl.ERROR_CODE.SUCCESS:
            print("[ERROR] No se pudo activar positional tracking."); zed.close(); sys.exit(1)
        print("[OK] Tracking activo: cajas en coords. de MUNDO. "
              "(No se saltan frames: el seek rompe el tracking)", flush=True)
    elif args.frame >= 0:
        zed.set_svo_position(min(args.frame, total - 1))

    rt = sl.RuntimeParameters()
    rt.confidence_threshold = args.zed_conf          # filtra profundidad dudosa (brillos)
    rt.texture_confidence_threshold = args.zed_texconf  # filtra saturado/sin textura
    img_mat, pc_mat, pose = sl.Mat(), sl.Mat(), sl.Pose()

    filas = []      # filas del CSV
    n = 0
    try:
        while True:
            estado = zed.grab(rt)
            if estado == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                print("[OK] Fin del SVO.", flush=True); break
            if estado != sl.ERROR_CODE.SUCCESS:
                print(f"[WARN] grab -> {estado}; fin.", flush=True); break
            n += 1
            fidx = zed.get_svo_position()
            if args.frame < 0 and n % 200 == 0:
                print(f"   frame {n}/{total}  detecciones 3D acumuladas: {len(filas)}",
                      flush=True)

            # Pose (solo en modo mundo). Sin tracking OK no fusionamos ese frame.
            R_w, t_w = None, None
            if args.mundo:
                if zed.get_position(pose, sl.REFERENCE_FRAME.WORLD) != sl.POSITIONAL_TRACKING_STATE.OK:
                    if args.max_frames and n >= args.max_frames: break
                    continue
                t_w = pose.get_translation(sl.Translation()).get()
                q_w = pose.get_orientation(sl.Orientation()).get()
                R_w = quat_a_R(q_w)

            # ¿Toca correr YOLO en este frame?
            if args.frame < 0 and args.cada > 1 and (n - 1) % args.cada != 0:
                if args.max_frames and n >= args.max_frames: break
                continue

            zed.retrieve_image(img_mat, sl.VIEW.LEFT)
            zed.retrieve_measure(pc_mat, sl.MEASURE.XYZRGBA)
            rgb = img_mat.get_data()[:, :, :3].copy()          # BGR
            pc = pc_mat.get_data()                              # HxWx4
            H, W = rgb.shape[:2]

            # --- 1. YOLO sobre la imagen RGB ---
            res = model.predict(rgb, conf=args.conf, verbose=False)[0]
            vis2d = rgb.copy()
            geoms = []

            # Recolectar TODAS las cajas 2D del frame antes de procesar
            dets = []
            for box in res.boxes:
                clase = nombres[int(box.cls[0])]
                if args.clases and clase not in args.clases:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                dets.append({"clase": clase, "conf": float(box.conf[0]),
                             "caja": (x1, y1, x2, y2)})

            # MACRO primero: de cada grupo sobrepuesto queda 1 sola caja
            dets, descartadas = suprimir_solapadas(dets, args.solape,
                                                   args.solape_cruzado)
            if descartadas:
                print(f"[f{fidx}] {len(descartadas)} caja(s) pospuesta(s) por solape: "
                      + ", ".join(f"{d['clase']}({d['conf']:.2f})" for d in descartadas),
                      flush=True)
                if args.show or args.guardar_jpg:
                    for d in descartadas:
                        dx1, dy1, dx2, dy2 = d["caja"]
                        cv2.rectangle(vis2d, (dx1, dy1), (dx2, dy2), (130, 130, 130), 1)
                        cv2.putText(vis2d, f"{d['clase']} (solapada)",
                                    (dx1, max(15, dy1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 130), 1)

            for i, d in enumerate(dets):
                clase, confd = d["clase"], d["conf"]
                x1, y1, x2, y2 = d["caja"]
                color = PALETA[i % len(PALETA)]

                # Dibujar SIEMPRE la caja 2D del YOLO (aunque el 3D luego falle):
                # para revisar el etiquetado hay que ver lo que el detector dijo
                if args.show or args.guardar_jpg:
                    bgr = tuple(int(c * 255) for c in color[::-1])
                    cv2.rectangle(vis2d, (x1, y1), (x2, y2), bgr, 2)
                    cv2.putText(vis2d, f"{clase} {confd:.2f}", (x1, max(15, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 2)

                # --- 2. Recorte de la nube organizada con la caja 2D ---
                crop = pc[y1:y2, x1:x2, :3]
                Hc, Wc = crop.shape[:2]
                xyz = crop.reshape(-1, 3)
                col = (rgb[y1:y2, x1:x2, ::-1] / 255.0).reshape(-1, 3)
                # coords de píxel de cada punto dentro del recorte (para saber
                # cuáles caen en el centro de la caja 2D)
                vv, uu = np.mgrid[0:Hc, 0:Wc]
                vv, uu = vv.ravel(), uu.ravel()
                fin = np.isfinite(xyz).all(axis=1)
                xyz, col, vv, uu = xyz[fin], col[fin], vv[fin], uu[fin]
                if len(xyz) < args.min_obj_pts:
                    continue
                dist = np.linalg.norm(xyz, axis=1)
                cerca = dist < args.max_dist
                xyz, col, vv, uu = xyz[cerca], col[cerca], vv[cerca], uu[cerca]
                if len(xyz) < args.min_obj_pts:
                    continue

                # Anti-oclusión: quedarnos con la capa de profundidad del objeto
                # (descarta el oclusor de adelante y el fondo de atrás)
                xyz, col = filtrar_banda_central(xyz, col, vv, uu, Hc, Wc, args.banda)
                if len(xyz) < args.min_obj_pts:
                    continue

                # Cámara -> mundo (si aplica) ANTES de limpiar, para que la OBB
                # salga directamente en coordenadas del mapa
                normal_fb = None
                if args.mundo:
                    xyz = xyz @ R_w.T + t_w
                    normal_fb = np.array([0.0, 0.0, 1.0])   # 'arriba' del mundo Z_UP

                # --- 3. Limpieza + OBB (lógica de limpieza_obb.py) ---
                objeto, obb = limpiar_y_obb(xyz, col, args, normal_fallback=normal_fb)
                if obb is None:
                    continue

                c, e = obb.center, obb.extent
                yaw = float(np.degrees(np.arctan2(obb.R[1, 0], obb.R[0, 0])))
                filas.append([fidx, clase, round(confd, 3),
                              round(c[0], 3), round(c[1], 3), round(c[2], 3),
                              round(e[0], 3), round(e[1], 3), round(e[2], 3),
                              round(yaw, 1), len(objeto.points),
                              "mundo" if args.mundo else "camara"])
                print(f"[f{fidx}] {clase} ({confd:.2f})  "
                      f"centro=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})m  "
                      f"dims={e[0]:.2f}x{e[1]:.2f}x{e[2]:.2f}m  pts={len(objeto.points)}",
                      flush=True)

                if args.guardar_ply:
                    ruta = os.path.join(DIR, f"objeto_f{fidx:04d}_{i}.ply")
                    o3d.io.write_point_cloud(ruta, objeto)

                if args.show:
                    objeto.paint_uniform_color(color)   # objeto y su caja, mismo color
                    obb.color = color
                    geoms += [objeto, obb]

            if args.guardar_jpg and (dets or descartadas):
                os.makedirs(JPG_DIR, exist_ok=True)
                cv2.imwrite(os.path.join(JPG_DIR, f"frame_{fidx:05d}.jpg"), vis2d)

            if args.show:
                cv2.imshow("YOLO 2D (frame %d)" % fidx, vis2d)
                cv2.waitKey(1 if geoms else 300)
                if geoms:
                    # Escena completa ATENUADA como contexto, para que los objetos
                    # (colores vivos) y sus cajas no queden flotando en el vacío
                    ctx_xyz = pc[::3, ::3, :3].reshape(-1, 3)
                    ctx_rgb = (rgb[::3, ::3, ::-1] / 255.0).reshape(-1, 3) * 0.35
                    fin = np.isfinite(ctx_xyz).all(axis=1)
                    ctx_xyz, ctx_rgb = ctx_xyz[fin], ctx_rgb[fin]
                    if args.mundo:
                        ctx_xyz = ctx_xyz @ R_w.T + t_w
                    ctx = o3d.geometry.PointCloud()
                    ctx.points = o3d.utility.Vector3dVector(ctx_xyz.astype(np.float64))
                    ctx.colors = o3d.utility.Vector3dVector(ctx_rgb.astype(np.float64))
                    print("[INFO] Visor 3D: escena atenuada + objetos en color + OBBs. "
                          "'Q' para seguir.")
                    o3d.visualization.draw_geometries(
                        [ctx] + geoms, window_name=f"Cajas 3D frame {fidx}")
                cv2.destroyAllWindows()

            if args.frame >= 0:          # modo un-solo-frame
                break
            if args.max_frames and n >= args.max_frames:
                print(f"[OK] Límite de {args.max_frames} frames.", flush=True)
                break
    except Exception as e:
        import traceback
        print("[EXCEPCIÓN]", e, flush=True); traceback.print_exc()

    if args.mundo:
        zed.disable_positional_tracking()
    zed.close()

    # --- CSV de salida ---
    if filas:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "clase", "conf", "cx", "cy", "cz",
                        "dx", "dy", "dz", "yaw_deg", "n_pts", "marco"])
            w.writerows(filas)
        print(f"\n[OK] {len(filas)} detecciones 3D -> {args.csv}")
        print("     (dimensiones = estimación de mapeo, ~10-15% de error)")
    if args.guardar_jpg:
        print(f"[OK] Frames anotados en {JPG_DIR}")
    else:
        print("\n[WARN] No hubo detecciones 3D. Prueba bajar --conf o revisa el frame.")


if __name__ == "__main__":
    main()
