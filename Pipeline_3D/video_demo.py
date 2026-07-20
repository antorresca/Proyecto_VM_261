"""
Video demo — frame etiquetado (2D) + mapa 3D con cajas, lado a lado (Módulo P1)
-------------------------------------------------------------------------------
Genera un MP4 para la demo: a la IZQUIERDA el video real con las cajas 2D de
YOLO (los JPG de --guardar-jpg), a la DERECHA el mapa 3D con las cajas
consolidadas, la trayectoria de la cámara y un marcador que avanza
sincronizado con el frame de la izquierda.

El mapa se renderiza por proyección propia (numpy + OpenCV, painter's
algorithm) — SIN OpenGL, así que corre igual en el PC del lab por AnyDesk.

Necesita (todo sale del pipeline normal):
  - mapa_limpio.ply                      (mapeo_manual + limpiar_mapa)
  - objetos_consolidados.csv CON yaw_deg (consolidar_detecciones.py actualizado)
  - mapa_poses.csv                       (lo guarda mapeo_manual.py)
  - carpeta etiquetado\\ con frame_XXXXX.jpg (deteccion_obb.py --guardar-jpg)

Uso:
    python video_demo.py                      # defaults en ~\\Proyecto_ZED_P1
    python video_demo.py --muestra            # SOLO guarda demo_muestra.png (rápido,
                                              #  para revisar encuadre antes del render)
    python video_demo.py --fps 15 --sin-orbita --out demo_clase.mp4
    python video_demo.py --voxel-video 0.03   # mapa más liviano si va lento
"""

import os
import re
import csv
import argparse
import numpy as np
import cv2

DIR = os.path.join(os.path.expanduser("~"), "Proyecto_ZED_P1")

PALETA = [(0.90, 0.10, 0.10), (0.10, 0.60, 0.95), (0.10, 0.80, 0.30),
          (0.95, 0.75, 0.10), (0.80, 0.20, 0.80), (0.95, 0.45, 0.10),
          (0.20, 0.85, 0.85), (0.60, 0.40, 0.20)]

ARISTAS = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
           (0, 4), (1, 5), (2, 6), (3, 7)]


def ruta_abs(r):
    return r if os.path.isabs(r) else os.path.join(DIR, r)


def cargar_objetos(ruta):
    """Cajas consolidadas 'ok' con yaw. Devuelve lista de dicts."""
    cajas = []
    with open(ruta, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if "yaw_deg" not in (rd.fieldnames or []):
            print("[WARN] El CSV consolidado no tiene columna yaw_deg (consolidar "
                  "viejo). Uso yaw=0: las cajas saldrán sin rotar.\n"
                  "       Re-corre consolidar_detecciones.py actualizado para "
                  "orientarlas bien.")
        for r in rd:
            if r["estado"] != "ok":
                continue
            cajas.append({"id": int(r["id"]), "clase": r["clase"],
                          "c": np.array([float(r["cx"]), float(r["cy"]), float(r["cz"])]),
                          "d": np.array([float(r["d90x"]), float(r["d90y"]), float(r["d90z"])]),
                          "yaw": np.radians(float(r.get("yaw_deg", 0) or 0))})
    return cajas


def esquinas_obb(c, d, yaw):
    """8 esquinas de la caja (yaw alrededor de Z)."""
    dx, dy, dz = d / 2
    base = np.array([[sx * dx, sy * dy, sz * dz]
                     for sz in (-1, 1) for sx, sy in
                     [(-1, -1), (1, -1), (1, 1), (-1, 1)]])
    ca, sa = np.cos(yaw), np.sin(yaw)
    R = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    return base @ R.T + c


def camara_lookat(eye, target):
    """Ejes de una cámara virtual mirando de eye a target (mundo Z_UP)."""
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    up = np.array([0.0, 0.0, 1.0])
    der = np.cross(fwd, up)
    n = np.linalg.norm(der)
    if n < 1e-6:                      # mirando exactamente en vertical
        der = np.array([1.0, 0.0, 0.0]); n = 1.0
    der = der / n
    aba = np.cross(fwd, der)          # 'abajo' de la imagen
    return np.stack([der, aba, fwd])  # filas: x_img, y_img, z_prof


def proyectar(P, eye, Rc, f, cx, cy):
    """Mundo -> píxel. Devuelve (u, v, z) con z = profundidad (m)."""
    q = (P - eye) @ Rc.T
    z = q[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cx + f * q[:, 0] / z
        v = cy + f * q[:, 1] / z
    return u, v, z


def render_mapa(pts, cols, eye, Rc, f, W, H):
    """Panel 3D: puntos del mapa con painter's algorithm (lejos primero)."""
    u, v, z = proyectar(pts, eye, Rc, f, W / 2, H / 2)
    m = (z > 0.15) & np.isfinite(u) & np.isfinite(v)
    u, v, z, cs = u[m].astype(int), v[m].astype(int), z[m], cols[m]
    m2 = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z, cs = u[m2], v[m2], z[m2], cs[m2]
    orden = np.argsort(-z)            # lejos primero; lo cercano sobreescribe
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[v[orden], u[orden]] = np.clip(cs[orden][:, ::-1] * 255 * 1.35, 0, 255)
    return cv2.dilate(img, np.ones((2, 2), np.uint8))


def preparar_vista_superior(pts, cols, cajas, lado, z_corte):
    """Fondo ESTÁTICO del panel cenital (mapa + cajas): se calcula una sola vez;
    por frame solo se dibuja la trayectoria y el marcador de la cámara.
    Devuelve (imagen_fondo, funcion mundo->pixel)."""
    m = pts[:, 2] < z_corte if z_corte else np.ones(len(pts), dtype=bool)
    p, c = pts[m], cols[m]
    x0, x1 = np.percentile(p[:, 0], [1, 99])
    y0, y1 = np.percentile(p[:, 1], [1, 99])
    x0 -= 0.4; x1 += 0.4; y0 -= 0.4; y1 += 0.4
    W = H = lado
    esc = min(W / (x1 - x0), H / (y1 - y0))
    offx = (W - (x1 - x0) * esc) / 2
    offy = (H - (y1 - y0) * esc) / 2

    def a_px(x, y):
        return (int(offx + (x - x0) * esc), int(H - offy - (y - y0) * esc))

    img = np.zeros((H, W, 3), dtype=np.uint8)
    orden = np.argsort(p[:, 2])               # lo alto se dibuja de último
    u = (offx + (p[:, 0] - x0) * esc).astype(int)[orden]
    v = (H - offy - (p[:, 1] - y0) * esc).astype(int)[orden]
    cs = np.clip(c[orden][:, ::-1] * 255 * 1.35, 0, 255).astype(np.uint8)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    img[v[ok], u[ok]] = cs[ok]
    img = cv2.dilate(img, np.ones((2, 2), np.uint8))

    for cj in cajas:                          # cajas (rectángulo rotado + id)
        cx, cy = cj["c"][0], cj["c"][1]
        dx, dy = cj["d"][0] / 2, cj["d"][1] / 2
        ca, sa = np.cos(cj["yaw"]), np.sin(cj["yaw"])
        esq = [a_px(cx + ex * ca - ey * sa, cy + ex * sa + ey * ca)
               for ex, ey in [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]]
        bgr = tuple(int(x * 255) for x in PALETA[cj["id"] % len(PALETA)][::-1])
        cv2.polylines(img, [np.array(esq)], True, bgr, 2, cv2.LINE_AA)
        pos = a_px(cx, cy)
        cv2.putText(img, f"#{cj['id']}", (pos[0] - 12, pos[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, f"#{cj['id']}", (pos[0] - 12, pos[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1, cv2.LINE_AA)
    return img, a_px


def dibujar_caja(img, caja, eye, Rc, f, W, H):
    esq = esquinas_obb(caja["c"], caja["d"], caja["yaw"])
    u, v, z = proyectar(esq, eye, Rc, f, W / 2, H / 2)
    if (z < 0.15).any():
        return
    pix = np.stack([u, v], axis=1).astype(int)
    bgr = tuple(int(x * 255) for x in PALETA[caja["id"] % len(PALETA)][::-1])
    for a, b in ARISTAS:
        cv2.line(img, tuple(pix[a]), tuple(pix[b]), bgr, 2, cv2.LINE_AA)
    top = pix[4:].mean(axis=0).astype(int)      # centro de la cara superior
    txt = f"#{caja['id']} {caja['clase']}"
    cv2.putText(img, txt, (top[0] - 35, top[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (top[0] - 35, top[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, bgr, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Video demo: 2D etiquetado + mapa 3D")
    ap.add_argument("--mapa", default="mapa_limpio.ply")
    ap.add_argument("--obj", default="objetos_consolidados.csv")
    ap.add_argument("--poses", default="mapa_poses.csv")
    ap.add_argument("--jpgs", default="etiquetado",
                    help="Carpeta con frame_XXXXX.jpg de --guardar-jpg")
    ap.add_argument("--out", default="demo.mp4")
    ap.add_argument("--fps", type=int, default=10,
                    help="FPS del video (10 con --cada 6 a 60fps ~ tiempo real)")
    ap.add_argument("--lado", type=int, default=720, help="Alto de los paneles (px)")
    ap.add_argument("--voxel-video", type=float, default=0.0,
                    help="Downsample extra del mapa solo para el video (m). "
                         "0.03 si el render va lento.")
    ap.add_argument("--sin-orbita", action="store_true",
                    help="Vista 3D fija (por defecto orbita lentamente)")
    ap.add_argument("--z-corte-arriba", type=float, default=2.2,
                    help="En el panel cenital: ocultar puntos del mapa por encima "
                         "de esta altura (m) para que techo/lámparas no tapen la "
                         "planta. 0 = mostrar todo.")
    ap.add_argument("--muestra", action="store_true",
                    help="Solo genera demo_muestra.png con el primer frame "
                         "compuesto (para revisar encuadre sin esperar el render)")
    args = ap.parse_args()

    import open3d as o3d               # solo para LEER el .ply (sin visor)
    mapa = o3d.io.read_point_cloud(ruta_abs(args.mapa))
    if args.voxel_video > 0:
        mapa = mapa.voxel_down_sample(args.voxel_video)
    pts = np.asarray(mapa.points)
    cols = np.asarray(mapa.colors)
    if len(pts) == 0:
        print(f"[ERROR] No pude leer {args.mapa}"); return
    if len(cols) != len(pts):
        cols = np.full((len(pts), 3), 0.6)
    print(f"[OK] Mapa: {len(pts):,} puntos")

    cajas = cargar_objetos(ruta_abs(args.obj))
    print(f"[OK] {len(cajas)} cajas consolidadas (estado ok)")

    # Poses de la cámara (frame -> posición)
    poses = np.genfromtxt(ruta_abs(args.poses), delimiter=",", names=True)
    pose_f = poses["frame"].astype(int)
    pose_t = np.stack([poses["tx"], poses["ty"], poses["tz"]], axis=1)
    print(f"[OK] {len(pose_f)} poses de cámara")

    # JPGs etiquetados, ordenados por número de frame
    carpeta = ruta_abs(args.jpgs)
    jpgs = []
    for nom in sorted(os.listdir(carpeta)):
        m = re.match(r"frame_(\d+)\.jpe?g$", nom, re.IGNORECASE)
        if m:
            jpgs.append((int(m.group(1)), os.path.join(carpeta, nom)))
    if not jpgs:
        print(f"[ERROR] No hay frame_XXXXX.jpg en {carpeta}. Corre deteccion_obb "
              f"con --guardar-jpg."); return
    jpgs.sort()
    print(f"[OK] {len(jpgs)} frames etiquetados "
          f"({len(jpgs)/args.fps:.0f} s de video a {args.fps} fps)")

    # Cámara virtual del panel 3D: órbita elevada alrededor del centro del mapa
    centro = pts.mean(axis=0)
    ext = float(np.max(pts.max(axis=0)[:2] - pts.min(axis=0)[:2]))
    radio = max(3.0, 0.85 * ext)
    H = W = args.lado
    f_px = 0.95 * W

    print("[INFO] Preparando panel cenital...", flush=True)
    fondo_sup, a_px_sup = preparar_vista_superior(
        pts, cols, cajas, args.lado,
        args.z_corte_arriba if args.z_corte_arriba > 0 else None)

    if args.muestra:
        jpgs = jpgs[:1]

    vw = None
    if not args.muestra:
        salida = ruta_abs(args.out)
        vw = cv2.VideoWriter(salida, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (0, 0))  # tamaño real al primer frame
    frame_out = None

    for k, (fidx, ruta_jpg) in enumerate(jpgs):
        # --- panel derecho: mapa + cajas + trayectoria ---
        # órbita lenta: 30 grados en todo el video (sutil, no marea)
        ang = np.radians(35) if args.sin_orbita else np.radians(35 + 30 * k / max(1, len(jpgs) - 1))
        eye = centro + np.array([radio * np.cos(ang), radio * np.sin(ang), 0.75 * radio])
        Rc = camara_lookat(eye, centro)
        panel = render_mapa(pts, cols, eye, Rc, f_px, W, H)

        # trayectoria recorrida hasta este frame + posición actual
        hasta = pose_t[pose_f <= fidx]
        if len(hasta) > 1:
            u, v, z = proyectar(hasta, eye, Rc, f_px, W / 2, H / 2)
            m = z > 0.15
            tray = np.stack([u[m], v[m]], axis=1).astype(np.int32)
            if len(tray) > 1:
                cv2.polylines(panel, [tray], False, (80, 200, 255), 2, cv2.LINE_AA)
            if len(tray):
                cv2.circle(panel, tuple(tray[-1]), 8, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(panel, tuple(tray[-1]), 8, (0, 0, 0), 2, cv2.LINE_AA)

        for caja in cajas:
            dibujar_caja(panel, caja, eye, Rc, f_px, W, H)

        cv2.putText(panel, "Mapa 3D + objetos consolidados", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # --- panel cenital: fondo estático + trayectoria + cámara ---
        sup = fondo_sup.copy()
        if len(hasta):
            tr = np.array([a_px_sup(p[0], p[1]) for p in hasta], dtype=np.int32)
            if len(tr) > 1:
                cv2.polylines(sup, [tr], False, (80, 200, 255), 2, cv2.LINE_AA)
            cv2.circle(sup, tuple(tr[-1]), 7, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(sup, tuple(tr[-1]), 7, (0, 0, 0), 2, cv2.LINE_AA)
            if len(tr) > 5:                    # flecha de rumbo (movimiento reciente)
                d = tr[-1] - tr[-6]
                nrm = float(np.hypot(*d))
                if nrm > 3:
                    punta = (tr[-1] + (d / nrm * 22)).astype(int)
                    cv2.arrowedLine(sup, tuple(tr[-1]), tuple(punta),
                                    (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.5)
        cv2.putText(sup, "Vista superior (recorrido)", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # --- panel izquierdo: frame etiquetado ---
        izq = cv2.imread(ruta_jpg)
        if izq is None:
            continue
        esc = args.lado / izq.shape[0]
        izq = cv2.resize(izq, (int(izq.shape[1] * esc), args.lado))
        cv2.putText(izq, f"YOLO 2D  |  frame {fidx}", (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        frame_out = np.hstack([izq, panel, sup])
        if frame_out.shape[1] % 2:                       # ancho par para el codec
            frame_out = frame_out[:, :-1]

        if args.muestra:
            png = ruta_abs("demo_muestra.png")
            cv2.imwrite(png, frame_out)
            print(f"[OK] Muestra -> {png}  (revisa encuadre; luego corre sin --muestra)")
            return

        if vw is not None and not vw.isOpened():
            vw = cv2.VideoWriter(ruta_abs(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.fps, (frame_out.shape[1], frame_out.shape[0]))
            if not vw.isOpened():                        # fallback a AVI
                alt = os.path.splitext(ruta_abs(args.out))[0] + ".avi"
                vw = cv2.VideoWriter(alt, cv2.VideoWriter_fourcc(*"XVID"),
                                     args.fps, (frame_out.shape[1], frame_out.shape[0]))
                print(f"[WARN] mp4v no disponible; escribo {alt}")
        vw.write(frame_out)
        if (k + 1) % 25 == 0:
            print(f"   {k + 1}/{len(jpgs)} frames renderizados", flush=True)

    if vw is not None:
        vw.release()
        print(f"\n[OK] Video -> {ruta_abs(args.out)}")
        print("     Tip: --muestra genera solo un PNG para revisar encuadre; "
              "--voxel-video 0.03 acelera; --sin-orbita fija la vista.")


if __name__ == "__main__":
    main()
