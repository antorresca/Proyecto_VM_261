"""
Limpieza del mapa fusionado — quita flotantes de ventanas/luz (Módulo P1)
-------------------------------------------------------------------------
Los puntos blancos que flotan en mitad de la sala son "flying pixels": la luz de
las ventanas satura el sensor y la ZED calcula profundidad falsa. Son puntos
AISLADOS, así que un filtro por radio los borra sin tocar las superficies reales.

Pasos:
  1. radius_outlier_removal  -> elimina puntos con pocos vecinos cerca (flotantes).
  2. statistical_outlier_removal -> pule el ruido fino restante.
  3. (opcional --brillo) quita además puntos casi blancos (saturados) que queden.

Uso:
    python limpiar_mapa.py                       # limpia mapa_manual.ply
    python limpiar_mapa.py --radio 0.05 --min 16 # más/menos agresivo
    python limpiar_mapa.py --brillo 0.97         # además borra puntos casi blancos
    python limpiar_mapa.py "C:\\ruta\\otro.ply"
"""

import os
import sys
import argparse
import numpy as np
import open3d as o3d

DIR = os.path.join(os.path.expanduser("~"), "Proyecto_ZED_P1")
DEFECTO = os.path.join(DIR, "mapa_manual.ply")


def main():
    ap = argparse.ArgumentParser(description="Limpia flotantes de ventanas/luz del mapa")
    ap.add_argument("ply", nargs="?", default=DEFECTO)
    ap.add_argument("--radio", type=float, default=0.05,
                    help="Radio (m) en que se cuentan vecinos. Default 5cm.")
    ap.add_argument("--min", type=int, default=16,
                    help="Mínimo de vecinos dentro del radio para conservar un punto.")
    ap.add_argument("--std", type=float, default=2.0,
                    help="std_ratio del filtro estadístico (menor = más estricto).")
    ap.add_argument("--brillo", type=float, default=0.0,
                    help="Si >0, borra puntos con los 3 canales RGB por encima de este valor (0-1). Ej 0.97.")
    ap.add_argument("--sin-visor", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.ply):
        print(f"[ERROR] No existe: {args.ply}"); sys.exit(1)

    pcd = o3d.io.read_point_cloud(args.ply)
    N0 = len(pcd.points)
    print(f"[OK] Cargado: {args.ply}  ({N0:,} puntos)")
    if N0 == 0:
        sys.exit(1)

    # 1) Flotantes aislados (el principal culpable de las ventanas)
    print(f"[1/3] Radius outlier removal (radio={args.radio} m, min={args.min} vecinos)...")
    pcd, keep = pcd.remove_radius_outlier(nb_points=args.min, radius=args.radio)
    print(f"      -> quitados {N0 - len(keep):,}  | quedan {len(keep):,}")

    # 2) Ruido fino restante
    print(f"[2/3] Statistical outlier removal (std_ratio={args.std})...")
    n1 = len(pcd.points)
    pcd, keep = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=args.std)
    print(f"      -> quitados {n1 - len(keep):,}  | quedan {len(keep):,}")

    # 3) (opcional) puntos casi blancos / saturados
    if args.brillo > 0 and len(pcd.colors) == len(pcd.points):
        cols = np.asarray(pcd.colors)
        blancos = np.all(cols > args.brillo, axis=1)
        n2 = len(pcd.points)
        idx = np.where(~blancos)[0]
        pcd = pcd.select_by_index(idx)
        print(f"[3/3] Filtro de brillo (>{args.brillo}): quitados {n2 - len(idx):,}  | quedan {len(idx):,}")
    else:
        print("[3/3] Filtro de brillo: omitido (usa --brillo 0.97 para activarlo).")

    Nf = len(pcd.points)
    out = os.path.join(os.path.dirname(args.ply), "mapa_limpio.ply")
    o3d.io.write_point_cloud(out, pcd)
    print(f"\n[OK] Mapa limpio guardado: {out}")
    print(f"     {N0:,} -> {Nf:,} puntos  (quitados {N0 - Nf:,} = {100*(N0-Nf)/N0:.1f}%)")

    if not args.sin_visor:
        print("[INFO] Visor del mapa LIMPIO. 'Q' cierra.")
        ejes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        o3d.visualization.draw_geometries([pcd, ejes], window_name="Mapa limpio")


if __name__ == "__main__":
    main()
