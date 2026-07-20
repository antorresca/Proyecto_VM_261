"""
Vista de lado (alzado) — diagnóstico mapa vs detecciones — Módulo P1
--------------------------------------------------------------------
Responde: "¿por qué las cajas flotan?" y "¿dónde quedó el piso/techo?"

Dibuja el mapa proyectado de LADO (eje horizontal X o Y, vertical Z) con los
centros de las detecciones encima, y marca dos líneas:
  - ROJA:  z = 0 (donde consolidar_detecciones.py ASUME que está el piso
           para --al-piso, --z-min y --z-max)
  - VERDE: piso real detectado en el mapa (percentil 1 de las Z)

Si las dos líneas no coinciden, el problema NO es YOLO: el origen del mundo
quedó a la altura donde arrancó la cámara, no en el piso, y los flags de
consolidación están descalibrados para esa grabación. El script imprime el
desfase y los flags corregidos listos para copiar/pegar.

También avisa si las detecciones caen FUERA del volumen del mapa (señal de
que el .ply y el .csv son de corridas/grabaciones distintas).

Uso:
    python vista_lado.py mapa_lab.ply det_lab.csv
    python vista_lado.py mapa_lab.ply det_lab.csv --eje y --out alzado_lab
(rutas relativas se buscan en ~\\Proyecto_ZED_P1)
"""

import os
import csv
import argparse
import numpy as np

DIR = os.path.join(os.path.expanduser("~"), "Proyecto_ZED_P1")

PALETA = [(0.90, 0.10, 0.10), (0.10, 0.60, 0.95), (0.10, 0.80, 0.30),
          (0.95, 0.75, 0.10), (0.80, 0.20, 0.80), (0.95, 0.45, 0.10),
          (0.20, 0.85, 0.85), (0.60, 0.40, 0.20)]


def ruta_abs(r):
    return r if os.path.isabs(r) else os.path.join(DIR, r)


def cargar_det(ruta):
    dets = []
    with open(ruta, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            dets.append({"clase": r["clase"],
                         "c": np.array([float(r["cx"]), float(r["cy"]),
                                        float(r["cz"])])})
    return dets


def alzado(pts, cols, dets, eje, piso, ruta_png, ancho=1600):
    import cv2
    h = 0 if eje == "x" else 1          # eje horizontal del dibujo
    x0, x1 = np.percentile(pts[:, h], [1, 99])
    z0, z1 = np.percentile(pts[:, 2], [1, 99])
    x0 -= 0.4; x1 += 0.4
    z0 = min(z0, min((d["c"][2] for d in dets), default=z0), 0) - 0.4
    z1 = max(z1, max((d["c"][2] for d in dets), default=z1), 0) + 0.4
    esc = ancho / (x1 - x0)
    alto = max(200, int((z1 - z0) * esc))

    def a_px(hv, zv):
        return (int((hv - x0) * esc), int(alto - (zv - z0) * esc))

    img = np.zeros((alto + 1, ancho + 1, 3), dtype=np.uint8)
    u = ((pts[:, h] - x0) * esc).astype(int)
    v = (alto - (pts[:, 2] - z0) * esc).astype(int)
    m = (u >= 0) & (u <= ancho) & (v >= 0) & (v <= alto)
    if cols is not None and len(cols) == len(pts):
        img[v[m], u[m]] = np.clip(cols[m][:, ::-1] * 255 * 1.4, 0, 255).astype(np.uint8)
    else:
        img[v[m], u[m]] = 150
    img = cv2.dilate(img, np.ones((2, 2), np.uint8))

    # línea z=0 (ROJA) y piso detectado (VERDE)
    for zv, bgr, txt in [(0.0, (0, 0, 255), "z=0 (piso ASUMIDO por consolidar)"),
                         (piso, (0, 255, 0), f"piso REAL del mapa (z={piso:+.2f} m)")]:
        p1 = a_px(x0, zv); p2 = a_px(x1, zv)
        cv2.line(img, p1, p2, bgr, 2)
        cv2.putText(img, txt, (10, p1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 4)
        cv2.putText(img, txt, (10, p1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, bgr, 2)

    # centros de detecciones por clase
    clases = sorted(set(d["clase"] for d in dets))
    color_de = {c: PALETA[i % len(PALETA)] for i, c in enumerate(clases)}
    for d in dets:
        bgr = tuple(int(v * 255) for v in color_de[d["clase"]][::-1])
        cv2.circle(img, a_px(d["c"][h], d["c"][2]), 4, bgr, -1)
    # leyenda
    y = 30
    for c in clases:
        bgr = tuple(int(v * 255) for v in color_de[c][::-1])
        cv2.circle(img, (ancho - 180, y - 5), 6, bgr, -1)
        cv2.putText(img, c, (ancho - 165, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3)
        cv2.putText(img, c, (ancho - 165, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, bgr, 1)
        y += 24
    cv2.imwrite(ruta_png, img)


def main():
    ap = argparse.ArgumentParser(description="Alzado del mapa + detecciones (diagnóstico)")
    ap.add_argument("mapa", help="mapa .ply (p.ej. mapa_lab.ply)")
    ap.add_argument("csv", help="detecciones_3d.csv de la MISMA corrida")
    ap.add_argument("--eje", choices=["x", "y", "ambos"], default="ambos",
                    help="Eje horizontal del alzado (default: genera ambos)")
    ap.add_argument("--out", default="alzado", help="Prefijo de los PNG de salida")
    args = ap.parse_args()

    import open3d as o3d
    mapa = o3d.io.read_point_cloud(ruta_abs(args.mapa))
    pts = np.asarray(mapa.points)
    cols = np.asarray(mapa.colors)
    if len(pts) == 0:
        print(f"[ERROR] No pude leer {args.mapa}"); return
    dets = cargar_det(ruta_abs(args.csv))
    if not dets:
        print(f"[ERROR] {args.csv} vacío."); return

    piso = float(np.percentile(pts[:, 2], 1))
    techo = float(np.percentile(pts[:, 2], 99))
    cz = np.array([d["c"][2] for d in dets])

    print(f"[OK] Mapa: {len(pts):,} puntos | Detecciones: {len(dets)}")
    print(f"\n===== ALTURAS (eje Z del mundo) =====")
    print(f"  Piso del mapa (p1):     z = {piso:+.2f} m")
    print(f"  Tope del mapa (p99):    z = {techo:+.2f} m   "
          f"(si es < 2 m, el techo NO está en el mapa — "
          f"¿lo borró limpiar_mapa --brillo? mira mapa_manual.ply)")
    print(f"  Centros det.:  min {cz.min():+.2f} | mediana {np.median(cz):+.2f} "
          f"| max {cz.max():+.2f} m")
    por_clase = {}
    for d in dets:
        por_clase.setdefault(d["clase"], []).append(d["c"][2])
    for c, zs in sorted(por_clase.items()):
        print(f"    {c:<12} n={len(zs):<5} z mediana {np.median(zs):+.2f} m")

    # ¿mapa y csv son de la misma corrida?
    xy_map_min = pts[:, :2].min(axis=0) - 1.0
    xy_map_max = pts[:, :2].max(axis=0) + 1.0
    cxy = np.array([d["c"][:2] for d in dets])
    fuera = ((cxy < xy_map_min) | (cxy > xy_map_max)).any(axis=1).sum()
    if fuera > len(dets) * 0.2:
        print(f"\n[ALERTA] {fuera}/{len(dets)} detecciones caen FUERA del área "
              f"del mapa en XY.\n  El .ply y el .csv parecen de CORRIDAS O "
              f"GRABACIONES DISTINTAS — compara con archivos de la misma corrida.")

    print(f"\n===== VEREDICTO =====")
    if abs(piso) <= 0.15:
        print("  El piso del mapa SÍ está en z≈0. Los flags de consolidar "
              "aplican tal cual.\n  Si aun así las cajas flotan, revisa la "
              "ALERTA de arriba (archivos de corridas distintas).")
    else:
        print(f"  El piso del mapa está en z = {piso:+.2f} m, NO en 0: el origen "
              f"del mundo quedó\n  donde arrancó la cámara "
              f"({-piso:.2f} m sobre el piso). Los flags con piso-en-0 están\n"
              f"  descalibrados: --al-piso extruye hasta z=0 (en el aire), "
              f"--z-min bota o deja\n  pasar lo que no debe y --z-max corta mal. "
              f"Flags corregidos para ESTA corrida:\n")
        print(f"    --z-min {piso - 0.10:.2f} --z-max {piso + 2.0:.2f}   "
              f"(y NO usar --al-piso hasta ajustar consolidar)")
        print(f"\n  Arreglo de raíz para futuras corridas: en los parámetros de "
              f"tracking de\n  mapeo_manual.py y deteccion_obb.py activar "
              f"set_floor_as_origin=True\n  (PositionalTrackingParameters) — el "
              f"SDK pone el piso en z=0 solo, sin\n  importar a qué altura "
              f"arranque la cámara.")

    ejes = ["x", "y"] if args.eje == "ambos" else [args.eje]
    for e in ejes:
        png = os.path.join(DIR, f"{args.out}_{e}.png")
        alzado(pts, cols if len(cols) else None, dets, e, piso, png)
        print(f"\n[OK] Alzado (horizontal={e.upper()}, vertical=Z) -> {png}")
    print("\n  En el PNG: línea ROJA = z=0 asumido, línea VERDE = piso real. "
          "Si los puntos\n  de las detecciones quedan colgados entre ambas, ese "
          "es el desfase.")


if __name__ == "__main__":
    main()
