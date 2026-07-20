"""
Sprint F (plan B) — Mapeo multivista MANUAL (Módulo P1)
-------------------------------------------------------
El módulo Spatial Mapping del ZED SDK crashea en este equipo. Aquí lo evitamos:
usamos SOLO el positional tracking (pose de la cámara por frame) y construimos el
mapa nosotros mismos en Open3D:

  por cada frame:
    1. pose de la cámara en el mundo (positional tracking)
    2. nube de puntos del frame (en coords. de cámara)
    3. transformar la nube a coords. de MUNDO con la pose
    4. acumular; cada cierto número de frames, voxel-downsample para no crecer infinito

Además guarda las POSES (frame + traslación + cuaternión) de cada frame aceptado,
para poder anclar después las cajas 3D de objetos dentro del mapa global.

Resultado: nube fusionada del recorrido (mapa del lab) + poses cámara->mundo.

Uso:
    python mapeo_manual.py "C:\\ruta\\a\\tu.svo"
    python mapeo_manual.py "...tu.svo" --max-frames 150        # prueba rápida
    python mapeo_manual.py "...tu.svo" --sub 2 --voxel 0.015   # más denso
"""

import os
import sys
import argparse
import numpy as np
import pyzed.sl as sl
import open3d as o3d

DIR = os.path.join(os.path.expanduser("~"), "Proyecto_ZED_P1")
os.makedirs(DIR, exist_ok=True)
SALIDA = os.path.join(DIR, "mapa_manual.ply")
SALIDA_POSES = os.path.join(DIR, "mapa_poses.npz")   # poses cámara->mundo por frame
SALIDA_POSES_CSV = os.path.join(DIR, "mapa_poses.csv")


def quat_a_R(q):
    """Cuaternión (x,y,z,w) -> matriz de rotación 3x3."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def consolidar(pts, cols, voxel):
    """Une listas de puntos/colores en una nube y la reduce con voxel grid."""
    if not pts:
        return None
    p = np.vstack(pts)
    c = np.vstack(cols)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(p)
    pc.colors = o3d.utility.Vector3dVector(c)
    return pc.voxel_down_sample(voxel)


def main():
    ap = argparse.ArgumentParser(description="Mapeo multivista manual (sin Spatial Mapping)")
    ap.add_argument("svo")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = todos")
    ap.add_argument("--sub", type=int, default=4,
                    help="Submuestreo de píxeles (toma 1 de cada N en X e Y). Más alto = más liviano.")
    ap.add_argument("--max-dist", type=float, default=3.5, help="Descarta puntos más lejos que esto (m)")
    ap.add_argument("--voxel", type=float, default=0.02, help="Tamaño de voxel del mapa (m)")
    ap.add_argument("--cada", type=int, default=40,
                    help="Consolidar (voxel-downsample) cada N frames aceptados")
    ap.add_argument("--depth", choices=["neural_light", "neural", "neural_plus"],
                    default="neural_light")
    ap.add_argument("--conf", type=int, default=50,
                    help="confidence_threshold ZED (1-100, MENOR = más estricto). Quita profundidad dudosa de ventanas/luz.")
    ap.add_argument("--texconf", type=int, default=100,
                    help="texture_confidence_threshold ZED (1-100, menor quita zonas sin textura/saturadas como ventanas).")
    ap.add_argument("--origen-camara", action="store_true",
                    help="Comportamiento viejo: origen del mundo donde arrancó la "
                         "cámara (por defecto ahora el origen es el PISO, z=0, "
                         "sin importar a qué altura se grabó).")
    args = ap.parse_args()

    DEPTH = {"neural_light": sl.DEPTH_MODE.NEURAL_LIGHT,
             "neural": sl.DEPTH_MODE.NEURAL,
             "neural_plus": sl.DEPTH_MODE.NEURAL_PLUS}[args.depth]

    init = sl.InitParameters()
    init.set_from_svo_file(args.svo)
    init.svo_real_time_mode = False
    init.depth_mode = DEPTH
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
    init.depth_maximum_distance = args.max_dist

    zed = sl.Camera()
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("[ERROR] No se pudo abrir el .svo"); sys.exit(1)
    total = zed.get_svo_number_of_frames()
    print(f"[OK] SVO abierto. Frames: {total}", flush=True)

    track_par = sl.PositionalTrackingParameters()
    if not args.origen_camara:
        # PISO como origen del mundo: el SDK detecta el plano del suelo y pone
        # z=0 AHÍ, no donde arrancó la cámara. Así --z-min/--z-max/--al-piso de
        # consolidar_detecciones.py valen tal cual para CUALQUIER altura de
        # grabación. (El suelo debe ser visible en los primeros frames.)
        track_par.set_floor_as_origin = True
        print("[OK] Origen del mundo = PISO (set_floor_as_origin). "
              "Usar el MISMO modo en deteccion_obb.py.", flush=True)
    if zed.enable_positional_tracking(track_par) != sl.ERROR_CODE.SUCCESS:
        print("[ERROR] No se pudo activar positional tracking."); zed.close(); sys.exit(1)
    print("[OK] Positional tracking activo. (NO usamos Spatial Mapping)", flush=True)

    rt = sl.RuntimeParameters()
    rt.confidence_threshold = args.conf            # filtra profundidad poco confiable (ventanas/luz)
    rt.texture_confidence_threshold = args.texconf  # filtra zonas sin textura / saturadas
    img = sl.Mat()
    pc_mat = sl.Mat()
    pose = sl.Pose()

    pts, cols = [], []
    poses = []     # (frame_svo, tx,ty,tz, qx,qy,qz,qw) de cada frame aceptado
    mapa = None
    n = 0          # frames leídos
    acc = 0        # frames aceptados (con pose OK)
    s = max(1, args.sub)

    print("[INFO] Fusionando... progreso cada 25 frames.", flush=True)
    try:
        while True:
            estado = zed.grab(rt)
            if estado == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                print(f"[OK] Fin del SVO. Leídos: {n}, aceptados: {acc}", flush=True)
                break
            if estado != sl.ERROR_CODE.SUCCESS:
                print(f"[WARN] grab -> {estado}; fin.", flush=True); break
            n += 1

            track_state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
            if track_state != sl.POSITIONAL_TRACKING_STATE.OK:
                # tracking aún inicializando o perdido: saltar este frame
                if n % 25 == 0:
                    print(f"   frame {n}/{total}  (tracking: {track_state}, sin fusionar)", flush=True)
                if args.max_frames and n >= args.max_frames:
                    break
                continue

            # Pose cámara->mundo
            t = pose.get_translation(sl.Translation()).get()      # (3,)
            q = pose.get_orientation(sl.Orientation()).get()      # (4,) x,y,z,w
            R = quat_a_R(q)

            # Nube + color del frame
            zed.retrieve_image(img, sl.VIEW.LEFT)
            zed.retrieve_measure(pc_mat, sl.MEASURE.XYZRGBA)
            xyz = pc_mat.get_data()[::s, ::s, :3].reshape(-1, 3)
            rgb = (img.get_data()[::s, ::s, :3][:, :, ::-1] / 255.0).reshape(-1, 3)

            finite = np.isfinite(xyz).all(axis=1)   # quitar inf/nan PRIMERO
            xyz, rgb = xyz[finite], rgb[finite]
            if len(xyz) == 0:
                continue
            dist = np.linalg.norm(xyz, axis=1)      # ya sin infinitos -> sin overflow
            keep = dist < args.max_dist
            xyz, rgb = xyz[keep], rgb[keep]
            if len(xyz) == 0:
                continue

            # Cámara -> mundo
            world = xyz @ R.T + t
            pts.append(world)
            cols.append(rgb)
            # Registrar la pose de ESTE frame (para anclar cajas 3D después)
            fidx = zed.get_svo_position()
            poses.append((fidx, t[0], t[1], t[2], q[0], q[1], q[2], q[3]))
            acc += 1

            if n % 25 == 0:
                tot_pts = sum(len(p) for p in pts) + (len(mapa.points) if mapa else 0)
                print(f"   frame {n}/{total}  aceptados={acc}  pts~{tot_pts:,}", flush=True)

            # Consolidar periódicamente para acotar memoria
            if acc % args.cada == 0:
                nuevo = consolidar(([np.asarray(mapa.points)] if mapa else []) + pts,
                                   ([np.asarray(mapa.colors)] if mapa else []) + cols,
                                   args.voxel)
                mapa, pts, cols = nuevo, [], []

            if args.max_frames and n >= args.max_frames:
                print(f"[OK] Límite de {args.max_frames} frames.", flush=True)
                break
    except Exception as e:
        import traceback
        print("[EXCEPCIÓN]", e, flush=True); traceback.print_exc()

    zed.disable_positional_tracking()
    zed.close()

    # Consolidación final
    final = consolidar(([np.asarray(mapa.points)] if mapa else []) + pts,
                       ([np.asarray(mapa.colors)] if mapa else []) + cols,
                       args.voxel)
    if final is None or len(final.points) == 0:
        print("[ERROR] No se acumuló nada (¿tracking nunca llegó a OK?).")
        return

    o3d.io.write_point_cloud(SALIDA, final)
    print(f"[OK] Mapa guardado en {SALIDA}  ({len(final.points):,} puntos)", flush=True)

    # Guardar poses cámara->mundo (frame, traslación, cuaternión xyzw)
    if poses:
        P = np.array(poses, dtype=np.float64)
        frames = P[:, 0].astype(np.int64)
        trans = P[:, 1:4]      # (N,3)
        quats = P[:, 4:8]      # (N,4) x,y,z,w
        np.savez(SALIDA_POSES, frames=frames, trans=trans, quats=quats)
        with open(SALIDA_POSES_CSV, "w") as f:
            f.write("frame,tx,ty,tz,qx,qy,qz,qw\n")
            for row in P:
                f.write(f"{int(row[0])},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f},"
                        f"{row[4]:.6f},{row[5]:.6f},{row[6]:.6f},{row[7]:.6f}\n")
        print(f"[OK] Poses guardadas: {SALIDA_POSES}  ({len(poses)} frames)", flush=True)
    else:
        print("[WARN] No se registraron poses (¿tracking nunca llegó a OK?).", flush=True)

    print("[INFO] Visor 3D del mapa fusionado. 'Q' para cerrar.")
    o3d.visualization.draw_geometries([final], window_name="Mapa manual (plan B)")


if __name__ == "__main__":
    main()
