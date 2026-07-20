"""
Prueba de humo — Módulo P1 (Geometría 3D y Mapeo)
-------------------------------------------------
Abre un archivo .svo grabado con la ZED2, toma UN frame y muestra:
  1. Imagen RGB (ventana OpenCV)
  2. Mapa de profundidad coloreado (ventana OpenCV)
  3. Nube de puntos 3D coloreada (visor Open3D)

Si las tres cosas aparecen, tu base está lista para arrancar el Sprint A.

Uso:
    python replay_svo.py "ruta\\a\\tu_grabacion.svo"
    python replay_svo.py "ruta\\a\\tu.svo" --frame 50      # salta al frame 50

Requisitos (en el venv zed3d):  pyzed, open3d, opencv-python, numpy
"""

import sys
import argparse
import numpy as np
import cv2
import pyzed.sl as sl
import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description="Prueba de humo replay .svo")
    parser.add_argument("svo", help="Ruta al archivo .svo")
    parser.add_argument("--frame", type=int, default=0,
                        help="Número de frame a inspeccionar (def. 0 = primero)")
    parser.add_argument("--max-dist", type=float, default=8.0,
                        help="Distancia máxima de profundidad en metros (def. 8)")
    args = parser.parse_args()

    # --- Configuración de apertura ---
    init = sl.InitParameters()
    init.set_from_svo_file(args.svo)
    init.svo_real_time_mode = False                 # offline, sin tiempo real
    init.depth_mode = sl.DEPTH_MODE.NEURAL          # NEURAL ya está optimizado en tu GPU
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP  # Z arriba (mapeo)
    init.depth_maximum_distance = args.max_dist

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[ERROR] No se pudo abrir el .svo: {status}")
        sys.exit(1)

    total = zed.get_svo_number_of_frames()
    print(f"[OK] SVO abierto. Frames totales: {total}")

    # Saltar al frame pedido
    if args.frame > 0:
        zed.set_svo_position(min(args.frame, total - 1))

    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()
    pcd_mat = sl.Mat()

    if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        print("[ERROR] No se pudo leer el frame.")
        zed.close()
        sys.exit(1)

    pos = zed.get_svo_position()
    print(f"[OK] Frame leído: {pos}")

    # --- Recuperar datos ---
    zed.retrieve_image(image, sl.VIEW.LEFT)          # RGB (BGRA)
    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)    # profundidad en metros
    zed.retrieve_measure(pcd_mat, sl.MEASURE.XYZRGBA)  # nube XYZ + color

    rgb = image.get_data()[:, :, :3]                 # BGRA -> BGR
    depth_np = depth.get_data()                      # HxW float32 (metros, con NaN)

    # --- Estadística rápida de profundidad ---
    valid = np.isfinite(depth_np)
    if valid.any():
        print(f"[INFO] Profundidad válida: {valid.mean()*100:.1f}% de píxeles | "
              f"min={np.nanmin(depth_np[valid]):.2f} m  max={np.nanmax(depth_np[valid]):.2f} m")

    # --- Mapa de profundidad coloreado para visualizar ---
    d = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
    d_norm = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)

    cv2.imshow("RGB (izquierda)", rgb)
    cv2.imshow("Profundidad", depth_color)
    print("[INFO] Cierra las ventanas o presiona una tecla para ver la nube 3D...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --- Nube de puntos -> Open3D ---
    pc = pcd_mat.get_data()                          # HxWx4 float32 (X,Y,Z,RGBA-empaquetado)
    xyz = pc[:, :, :3].reshape(-1, 3)

    # Color: tomamos el RGB de la imagen (más simple que desempaquetar el canal float)
    colors = rgb.reshape(-1, 3)[:, ::-1] / 255.0     # BGR -> RGB, normalizado 0..1

    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    colors = colors[finite]
    print(f"[OK] Nube con {len(xyz):,} puntos válidos.")

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    zed.close()

    print("[INFO] Abriendo visor 3D. Arrastra para rotar, scroll para zoom, 'Q' para cerrar.")
    o3d.visualization.draw_geometries([cloud], window_name="Nube de puntos ZED2")


if __name__ == "__main__":
    main()
