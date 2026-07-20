"""
Sprints C-E — Limpieza de nube + OBB (Módulo P1)
------------------------------------------------
Toma la nube recortada del Sprint B (objeto_recortado.ply, que todavía trae
mesa/fondo dentro del recuadro) y deja SOLO el objeto, con su caja 3D orientada.

Pasos:
  C. Statistical outlier removal + voxel downsample  (limpia ruido)
  D. RANSAC -> elimina el plano de soporte (mesa/piso)
  E. DBSCAN -> conserva el cluster más grande (el objeto)
     PCA/OBB -> caja 3D orientada (centro, orientación, extensión)

Las dimensiones que imprime son una ESTIMACIÓN derivada del mapa (~10-15% error),
coherente con el enfoque de mapeo, NO una medición certificada.

Uso:
    python limpieza_obb.py                         # usa objeto_recortado.ply
    python limpieza_obb.py mi_nube.ply
    python limpieza_obb.py --voxel 0.005 --plano 0.015 --eps 0.03
"""

import os
import argparse
import numpy as np
import open3d as o3d

# Misma carpeta de salida que recorte_mascara.py (en tu home, escribible)
DIR = os.path.join(os.path.expanduser("~"), "Proyecto_ZED_P1")


def obb_alineada_al_plano(pcd, normal):
    """OBB con el eje Z anclado a la normal del plano de soporte y el giro
    horizontal (yaw) calculado por PCA 2D. Mucho más estable que el PCA 3D
    cuando solo se ve una cara del objeto (un solo frame)."""
    pts = np.asarray(pcd.points)
    n = np.asarray(normal, dtype=float)
    n /= np.linalg.norm(n)

    # Dos ejes ortonormales sobre el plano
    ref = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(n, ref); u /= np.linalg.norm(u)
    v = np.cross(n, u)

    c = pts.mean(axis=0)
    p = pts - c
    pu, pv = p @ u, p @ v                       # coords en el plano

    # PCA 2D -> dirección horizontal principal
    cov = np.cov(np.vstack([pu, pv]))
    w, vec = np.linalg.eigh(cov)
    e1 = vec[:, int(np.argmax(w))]              # eje largo horizontal
    axis1 = e1[0] * u + e1[1] * v
    axis2 = np.cross(n, axis1)
    R = np.column_stack([axis1, axis2, n])
    if np.linalg.det(R) < 0:
        R[:, 1] = -R[:, 1]

    proj = p @ R                                # coords en los ejes de la caja
    mn, mx = proj.min(axis=0), proj.max(axis=0)
    extent = mx - mn
    center = c + R @ ((mn + mx) / 2)
    return o3d.geometry.OrientedBoundingBox(center, R, extent)


def main():
    ap = argparse.ArgumentParser(description="Limpieza de nube + OBB")
    ap.add_argument("ply", nargs="?", default=os.path.join(DIR, "objeto_recortado.ply"))
    ap.add_argument("--voxel", type=float, default=0.005, help="Tamaño de voxel (m)")
    ap.add_argument("--plano", type=float, default=0.015,
                    help="Umbral de distancia RANSAC para el plano (m)")
    ap.add_argument("--eps", type=float, default=0.03, help="DBSCAN: radio de vecindad (m)")
    ap.add_argument("--min-pts", type=int, default=20, help="DBSCAN: puntos mínimos")
    ap.add_argument("--no-plano", action="store_true", help="No quitar plano de soporte")
    args = ap.parse_args()

    if not os.path.isabs(args.ply):
        args.ply = os.path.join(DIR, args.ply)
    pcd = o3d.io.read_point_cloud(args.ply)
    if len(pcd.points) == 0:
        print(f"[ERROR] {args.ply} está vacío o no existe.")
        return
    print(f"[OK] Nube cargada: {len(pcd.points):,} puntos")

    # --- C. Downsample + outlier removal ---
    pcd = pcd.voxel_down_sample(voxel_size=args.voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"[C] Tras voxel + outlier removal: {len(pcd.points):,} puntos")

    # --- D. Quitar plano de soporte (RANSAC) y guardar su normal ---
    plane_normal = None
    if not args.no_plano and len(pcd.points) > 100:
        plano, inliers = pcd.segment_plane(distance_threshold=args.plano,
                                           ransac_n=3, num_iterations=1000)
        a, b, c, d = plano
        plane_normal = np.array([a, b, c])      # normal del soporte = "arriba"
        # Solo lo quitamos si el plano es una porción relevante (evita comerse el objeto)
        if 0.15 < len(inliers) / len(pcd.points) < 0.9:
            pcd = pcd.select_by_index(inliers, invert=True)
            print(f"[D] Plano de soporte eliminado ({len(inliers):,} puntos). "
                  f"Restan: {len(pcd.points):,}")
        else:
            print(f"[D] Plano detectado pero ignorado (proporción atípica: "
                  f"{len(inliers)/len(pcd.points):.0%})")
    else:
        print("[D] Se omitió la eliminación de plano.")

    # --- E. Clustering: conservar el objeto principal ---
    if len(pcd.points) == 0:
        print("[ERROR] No quedaron puntos tras la limpieza. Ajusta --plano o usa --no-plano.")
        return

    labels = np.array(pcd.cluster_dbscan(eps=args.eps, min_points=args.min_pts))
    if labels.max() < 0:
        print("[E] DBSCAN no formó clusters; uso toda la nube. Prueba subir --eps.")
        objeto = pcd
    else:
        # cluster más grande (ignorando ruido = etiqueta -1)
        cuenta = np.bincount(labels[labels >= 0])
        mayor = int(np.argmax(cuenta))
        objeto = pcd.select_by_index(np.where(labels == mayor)[0])
        print(f"[E] {labels.max()+1} clusters. Objeto = cluster más grande "
              f"({len(objeto.points):,} puntos).")

    # --- OBB: anclada al plano si lo tenemos; si no, mínima robusta ---
    if plane_normal is not None:
        obb = obb_alineada_al_plano(objeto, plane_normal)
        print("[OBB] Orientación anclada al plano de soporte (Z = normal, yaw por PCA 2D).")
    else:
        obb = objeto.get_minimal_oriented_bounding_box(robust=True)
        print("[OBB] Sin plano: usando caja mínima robusta (PCA 3D).")
    obb.color = (1, 0, 0)
    ext = obb.extent  # lados de la caja (m)
    print("\n===== OBJETO 3D (estimación de mapeo, no metrología) =====")
    print(f"  Centro (m):     X={obb.center[0]:.3f}  Y={obb.center[1]:.3f}  Z={obb.center[2]:.3f}")
    print(f"  Dimensiones (m): {ext[0]:.3f} x {ext[1]:.3f} x {ext[2]:.3f}")
    print(f"  (~10-15% de error esperado)")
    print("===========================================================\n")

    # Color uniforme al objeto para verlo claro junto a la caja
    objeto.paint_uniform_color([0.1, 0.6, 0.9])
    ejes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=obb.center)

    print("[INFO] Visor 3D: objeto limpio + caja orientada (roja). 'Q' para cerrar.")
    o3d.visualization.draw_geometries([objeto, obb, ejes],
                                      window_name="Objeto + OBB (Sprints C-E)")


if __name__ == "__main__":
    main()
