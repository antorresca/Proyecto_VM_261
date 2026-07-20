"""
Consolidación multivista — Módulo P1
------------------------------------
Responde la pregunta: "¿mejora al hacerlo con todo el recorrido?"

Lee detecciones_3d.csv (generado por deteccion_obb.py, idealmente con --mundo
para que todas las cajas estén en el mismo marco del mapa) y agrupa las
detecciones de todos los frames por OBJETO FÍSICO (misma clase + centros 3D
cercanos). Por cada objeto reporta:

  - en cuántos frames se vio (más frames = detección confiable)
  - centro robusto (mediana de los centros)
  - dimensiones consolidadas: como cada frame solo ve UNA cara del objeto,
    la caja por-frame tiende a quedarse CORTA; al juntar muchos ángulos, el
    percentil 90 de cada dimensión se acerca al tamaño real. Se reporta
    mediana (conservadora) y p90 (consolidada).

Los grupos con pocas detecciones (--min-det) se marcan como dudosos: suelen
ser falsos positivos que no se repiten entre frames.

Sigue siendo una ESTIMACIÓN derivada del mapeo (~10-15% error), no metrología.

Uso:
    python consolidar_detecciones.py
    python consolidar_detecciones.py --csv otra_ruta.csv --radio 0.6
    # ver las cajas consolidadas sobre el mapa global:
    python consolidar_detecciones.py --ver mapa_manual.ply
"""

import os
import csv
import argparse
import numpy as np

DIR = os.path.join(os.path.expanduser("~"), "Proyecto_ZED_P1")


def cargar(ruta):
    filas = []
    with open(ruta, newline="") as f:
        for r in csv.DictReader(f):
            filas.append({
                "frame": int(r["frame"]), "clase": r["clase"],
                "conf": float(r["conf"]),
                "c": np.array([float(r["cx"]), float(r["cy"]), float(r["cz"])]),
                "d": np.array([float(r["dx"]), float(r["dy"]), float(r["dz"])]),
                "yaw": float(r["yaw_deg"]), "n_pts": int(r["n_pts"]),
                "marco": r["marco"],
            })
    return filas


def agrupar(dets, radio):
    """Agrupación greedy: misma clase + centro a menos de `radio` (en el plano XY;
    la Z no cuenta para asociar, así los reflejos del piso y las vistas parciales
    del mismo objeto no crean grupos separados) -> mismo objeto físico."""
    grupos = []
    for d in sorted(dets, key=lambda x: -x["conf"]):
        for g in grupos:
            if g["clase"] == d["clase"] and np.linalg.norm((g["centro"] - d["c"])[:2]) < radio:
                g["items"].append(d)
                g["centro"] = np.median([i["c"] for i in g["items"]], axis=0)
                break
        else:
            grupos.append({"clase": d["clase"], "centro": d["c"].copy(), "items": [d]})

    # Pasada de fusión: al recalcular la mediana, dos grupos del mismo objeto
    # pueden quedar más cerca que `radio` — se unen hasta que no haya cambios
    fusionado = True
    while fusionado:
        fusionado = False
        for i in range(len(grupos)):
            for j in range(i + 1, len(grupos)):
                if (grupos[i]["clase"] == grupos[j]["clase"] and
                        np.linalg.norm((grupos[i]["centro"] - grupos[j]["centro"])[:2]) < radio):
                    grupos[i]["items"] += grupos[j]["items"]
                    grupos[i]["centro"] = np.median([it["c"] for it in grupos[i]["items"]], axis=0)
                    del grupos[j]
                    fusionado = True
                    break
            if fusionado:
                break
    return grupos


def vista_superior_png(mapa, cajas, ruta, ancho=1600):
    """Plano cenital (vista desde arriba) del mapa con las cajas y sus etiquetas,
    dibujado con OpenCV — sin GUI 3D, funciona en cualquier equipo.
    cajas = lista de dicts {id, clase, c (3,), d (3,), yaw (rad), color (r,g,b)}."""
    import cv2
    pts = np.asarray(mapa.points)
    cols = np.asarray(mapa.colors)
    if len(pts) == 0:
        return False
    x0, x1 = np.percentile(pts[:, 0], [1, 99])
    y0, y1 = np.percentile(pts[:, 1], [1, 99])
    x0 -= 0.4; x1 += 0.4; y0 -= 0.4; y1 += 0.4
    esc = ancho / (x1 - x0)
    alto = max(200, int((y1 - y0) * esc))

    def a_px(xy):
        return (int((xy[0] - x0) * esc), int(alto - (xy[1] - y0) * esc))

    img = np.zeros((alto + 1, ancho + 1, 3), dtype=np.uint8)
    u = ((pts[:, 0] - x0) * esc).astype(int)
    v = (alto - (pts[:, 1] - y0) * esc).astype(int)
    m = (u >= 0) & (u <= ancho) & (v >= 0) & (v <= alto)
    if len(cols) == len(pts):
        img[v[m], u[m]] = np.clip(cols[m][:, ::-1] * 255 * 1.4, 0, 255).astype(np.uint8)
    else:
        img[v[m], u[m]] = 170
    img = cv2.dilate(img, np.ones((2, 2), np.uint8))   # engordar puntos

    for cj in cajas:
        cx, cy = cj["c"][0], cj["c"][1]
        dx, dy = cj["d"][0] / 2, cj["d"][1] / 2
        ca, sa = np.cos(cj["yaw"]), np.sin(cj["yaw"])
        esquinas = []
        for ex, ey in [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]:
            esquinas.append(a_px((cx + ex * ca - ey * sa, cy + ex * sa + ey * ca)))
        bgr = tuple(int(c * 255) for c in cj["color"][::-1])
        cv2.polylines(img, [np.array(esquinas)], True, bgr, 2)
        etiqueta = f"#{cj['id']} {cj['clase']}"
        pos = a_px((cx, cy))
        cv2.putText(img, etiqueta, (pos[0] - 40, pos[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, etiqueta, (pos[0] - 40, pos[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)
    # ejes del mundo para orientarse (origen = arranque del recorrido)
    o = a_px((0, 0)); px_ = a_px((0.5, 0)); py = a_px((0, 0.5))
    cv2.arrowedLine(img, o, px_, (80, 80, 255), 2); cv2.putText(img, "X", px_, 0, 0.6, (80, 80, 255), 2)
    cv2.arrowedLine(img, o, py, (80, 255, 80), 2); cv2.putText(img, "Y", py, 0, 0.6, (80, 255, 80), 2)
    cv2.imwrite(ruta, img)
    return True


def ver_con_etiquetas(geoms, etiquetas, titulo):
    """Visor con texto 3D sobre cada caja (O3DVisualizer). Si la GUI nueva de
    Open3D falla en este equipo, cae al visor clásico (sin etiquetas)."""
    import open3d as o3d
    try:
        import open3d.visualization.gui as gui
        app = gui.Application.instance
        app.initialize()
        vis = o3d.visualization.O3DVisualizer(titulo, 1400, 900)
        vis.show_settings = False
        for j, g in enumerate(geoms):
            vis.add_geometry(f"g{j}", g)
        for pos, txt in etiquetas:
            vis.add_3d_label(pos, txt)
        vis.reset_camera_to_default()
        app.add_window(vis)
        app.run()
    except Exception as e:
        print(f"[WARN] Visor con etiquetas no disponible ({e}). Uso el visor simple;"
              f" guíate por la leyenda de colores impresa arriba.")
        o3d.visualization.draw_geometries(geoms, window_name=titulo)


def main():
    ap = argparse.ArgumentParser(description="Consolidar detecciones 3D del recorrido")
    ap.add_argument("--csv", default=os.path.join(DIR, "detecciones_3d.csv"))
    ap.add_argument("--radio", type=float, default=0.5,
                    help="Distancia máxima (m) entre centros para considerar "
                         "que dos detecciones son el mismo objeto")
    ap.add_argument("--min-det", type=int, default=3,
                    help="Detecciones mínimas para considerar el objeto confiable")
    ap.add_argument("--p", type=float, default=90,
                    help="Percentil para las dims consolidadas (bájalo a 75-80 si "
                         "aún se cuelan detecciones infladas por ruido)")
    ap.add_argument("--z-max", type=float, default=None,
                    help="Altura máxima (m, eje Z del mapa) del CENTRO de un objeto. "
                         "Los grupos más altos se marcan como luz/lámpara y no se "
                         "dibujan. Prueba 2.0 para un lab con techo a ~2.5-3 m.")
    ap.add_argument("--z-min", type=float, default=-0.10,
                    help="Descartar DETECCIONES con centro por debajo de esta altura "
                         "(reflejos del piso brillante: el objeto espejado bajo el "
                         "suelo). None/-99 para desactivar.")
    ap.add_argument("--sin-clases", nargs="*", default=None,
                    help="Clases a excluir de la consolidación (p.ej. --sin-clases ventana)")
    ap.add_argument("--z-corte", type=float, default=None,
                    help="En --ver: ocultar los puntos del mapa por encima de esta "
                         "altura (m). Corta techo y lámparas para ver las cajas.")
    ap.add_argument("--vista-limpia", action="store_true",
                    help="En --ver: outlier removal al mapa antes de mostrarlo "
                         "(quita nubes flotantes de ruido; tarda unos segundos)")
    ap.add_argument("--gui", action="store_true",
                    help="En --ver: intentar el visor 3D con etiquetas de texto "
                         "(falla en algunas GPUs; el PNG cenital sale siempre)")
    ap.add_argument("--forzar", action="store_true",
                    help="Continuar aunque el CSV esté en marco 'camara' (no recomendado)")
    ap.add_argument("--al-piso", nargs="*", default=None, metavar="CLASE",
                    help="Clases que se apoyan en el piso (p.ej. --al-piso silla mesa "
                         "estante): su altura se corrige extruyendo la caja hasta el "
                         "suelo (z=0) — compensa que la cámara baja solo ve la parte "
                         "superior. Solo aplica a grupos con centro < 1.2 m.")
    ap.add_argument("--out", default=os.path.join(DIR, "objetos_consolidados.csv"))
    ap.add_argument("--ver", metavar="MAPA_PLY",
                    help="Ruta a mapa_manual.ply para ver las cajas sobre el mapa")
    args = ap.parse_args()

    dets = cargar(args.csv)
    if not dets:
        print(f"[ERROR] {args.csv} está vacío."); return
    if args.sin_clases:
        antes = len(dets)
        dets = [d for d in dets if d["clase"] not in args.sin_clases]
        print(f"[OK] Excluidas clases {args.sin_clases}: {antes - len(dets)} detecciones fuera")
    if not dets:
        print("[ERROR] No quedaron detecciones tras excluir clases."); return
    if args.z_min is not None and args.z_min > -90:
        antes = len(dets)
        dets = [d for d in dets if d["c"][2] >= args.z_min]
        if antes - len(dets):
            print(f"[OK] Reflejos del piso descartados (centro < {args.z_min} m): "
                  f"{antes - len(dets)} detecciones")
    print(f"[OK] {len(dets)} detecciones de {len(set(d['frame'] for d in dets))} frames")

    marcos = set(d["marco"] for d in dets)
    if "camara" in marcos:
        print("\n[ERROR] Este CSV está en marco 'CAMARA' (la corrida fue SIN --mundo).\n"
              "  Cada frame tiene su propio origen (la cámara se mueve), así que\n"
              "  agrupar entre frames o dibujar sobre el mapa NO tiene sentido —\n"
              "  las cajas caerían en lugares aleatorios.\n"
              "  Re-corre:  python deteccion_obb.py <svo> --modelo best.pt --cada 3 --mundo\n"
              "  (--forzar para continuar bajo tu propio riesgo)")
        if not args.forzar:
            return

    grupos = agrupar(dets, args.radio)
    grupos.sort(key=lambda g: -len(g["items"]))

    print(f"\n===== {len(grupos)} OBJETOS FÍSICOS (radio {args.radio} m) =====")
    print("(dims = estimación de mapeo ~10-15% error; p90 consolida los ángulos)\n")
    salida = []
    for k, g in enumerate(grupos):
        its = g["items"]

        # Anti-brillo: descartar detecciones cuya caja quedó INFLADA respecto a
        # la mediana del grupo (típico de puntos de ruido por luz/reflejos)
        infladas = 0
        if len(its) >= 4:
            D0 = np.array([i["d"] for i in its])
            med0 = np.median(D0, axis=0)
            sano = (D0 <= np.maximum(med0 * 2.5, med0 + 0.30)).all(axis=1)
            if sano.sum() >= max(2, len(its) // 4):
                infladas = len(its) - int(sano.sum())
                its = [i for i, s in zip(its, sano) if s]
                g["items"] = its                # que --ver use también lo filtrado

        nf = len(set(i["frame"] for i in its))
        c = np.median([i["c"] for i in its], axis=0)
        D = np.array([i["d"] for i in its])            # (n,3) dims por frame
        d_med = np.median(D, axis=0)
        d_p90 = np.percentile(D, args.p, axis=0)

        # Extrusión al piso: para clases que se apoyan en el suelo, la altura
        # real es del TOPE del objeto al piso (z=0), no solo lo que la cámara
        # baja alcanzó a ver de la cara superior
        if (args.al_piso and g["clase"] in args.al_piso and c[2] < 1.2):
            topes = np.array([i["c"][2] + i["d"][2] / 2 for i in its])
            altura = float(np.percentile(topes, args.p))
            if altura > d_p90[2]:              # solo si mejora la estimación
                d_p90 = d_p90.copy(); d_med = d_med.copy()
                d_p90[2] = altura
                d_med[2] = max(d_med[2], float(np.median(topes)))
                c = c.copy(); c[2] = altura / 2   # centro a media altura real
                g["extruido"] = True
        conf = np.median([i["conf"] for i in its])
        g["alto"] = args.z_max is not None and c[2] > args.z_max
        dudoso = " [DUDOSO: pocas vistas]" if nf < args.min_det else ""
        if g["alto"]:
            dudoso += f" [ALTO: centro a {c[2]:.2f} m — ¿luz/lámpara?]"
        print(f"#{k} {g['clase']}  vistas={nf} frames ({len(its)} det., conf~{conf:.2f}){dudoso}")
        print(f"    centro (m): ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})")
        print(f"    dims mediana: {d_med[0]:.2f} x {d_med[1]:.2f} x {d_med[2]:.2f} m"
              f"   | dims p{args.p:.0f}: {d_p90[0]:.2f} x {d_p90[1]:.2f} x {d_p90[2]:.2f} m")
        if infladas:
            print(f"    ({infladas} det. descartadas por caja inflada — ruido de brillo/luz)")
        if g.get("extruido"):
            print("    (altura extruida hasta el piso — clase apoyada en el suelo)")
        estado = "ok" if nf >= args.min_det else "dudoso"
        if g["alto"]:
            estado = "alto_luz"
        yaw_med = float(np.median([i["yaw"] for i in its]))   # para video_demo.py
        salida.append([k, g["clase"], nf, len(its), round(conf, 2),
                       *np.round(c, 3), *np.round(d_med, 3), *np.round(d_p90, 3),
                       round(yaw_med, 1), estado])

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "clase", "frames_visto", "n_det", "conf_med",
                    "cx", "cy", "cz", "dmx", "dmy", "dmz",
                    "d90x", "d90y", "d90z", "yaw_deg", "estado"])
        w.writerows(salida)
    print(f"\n[OK] Resumen -> {args.out}")

    # --- Visualización opcional sobre el mapa global ---
    if args.ver:
        import open3d as o3d
        ruta = args.ver if os.path.isabs(args.ver) else os.path.join(DIR, args.ver)
        mapa = o3d.io.read_point_cloud(ruta)
        if len(mapa.points) == 0:
            print(f"[ERROR] No pude leer el mapa {ruta}"); return
        if args.z_corte is not None:
            pts = np.asarray(mapa.points)
            idx = np.where(pts[:, 2] < args.z_corte)[0]
            print(f"[VER] z-corte {args.z_corte} m: {len(pts) - len(idx):,} puntos "
                  f"de techo/lámparas ocultos")
            mapa = mapa.select_by_index(idx)
        if args.vista_limpia:
            antes = len(mapa.points)
            print("[VER] Outlier removal del mapa (puede tardar unos segundos)...")
            mapa, _ = mapa.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.6)
            print(f"[VER] Nubes flotantes eliminadas: {antes - len(mapa.points):,} puntos")
        # atenuar el mapa para que resalten las cajas
        cols = np.asarray(mapa.colors)
        if len(cols):
            mapa.colors = o3d.utility.Vector3dVector(cols * 0.45)
        geoms = [mapa]
        etiquetas = []
        cajas_png = []
        NOMBRES_COLOR = ["rojo", "azul", "verde", "amarillo", "morado", "naranja"]
        PALETA = [(0.90, 0.10, 0.10), (0.10, 0.60, 0.95), (0.10, 0.80, 0.30),
                  (0.95, 0.75, 0.10), (0.80, 0.20, 0.80), (0.95, 0.45, 0.10)]
        print("\n[VER] Leyenda de cajas dibujadas:")
        for k, g in enumerate(grupos):
            its = g["items"]
            nf = len(set(i["frame"] for i in its))
            if nf < args.min_det or g.get("alto"):
                continue                      # dudosas y luces no se dibujan
            c = np.median([i["c"] for i in its], axis=0)
            d = np.percentile([i["d"] for i in its], args.p, axis=0)
            yaw = np.radians(np.median([i["yaw"] for i in its]))
            R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                          [np.sin(yaw),  np.cos(yaw), 0],
                          [0, 0, 1]])
            color = PALETA[k % len(PALETA)]
            obb = o3d.geometry.OrientedBoundingBox(c, R, d)
            obb.color = color
            geoms.append(obb)
            etiquetas.append((c + np.array([0, 0, d[2] / 2 + 0.10]),
                              f"#{k} {g['clase']} ({nf}v)"))
            cajas_png.append({"id": k, "clase": g["clase"], "c": c, "d": d,
                              "yaw": yaw, "color": color})
            print(f"   caja #{k} [{NOMBRES_COLOR[k % 6]}] {g['clase']}  "
                  f"centro=({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})  vistas={nf}")

        # 1) PNG cenital con etiquetas — sale SIEMPRE, sin depender de la GPU
        png = os.path.join(DIR, "vista_arriba.png")
        if vista_superior_png(mapa, cajas_png, png):
            print(f"\n[OK] Plano cenital con etiquetas -> {png}")

        # 2) Visor 3D: clásico por defecto (sin etiquetas, usa la leyenda);
        #    con --gui se intenta el visor nuevo con texto 3D
        if args.gui:
            print("[INFO] Visor 3D con etiquetas (--gui). Cierra la ventana para terminar.")
            ver_con_etiquetas(geoms, etiquetas, "Objetos consolidados en el mapa")
        else:
            print("[INFO] Visor 3D clásico (identifica las cajas por color con la "
                  "leyenda de arriba o por el PNG cenital). 'Q' para cerrar.")
            o3d.visualization.draw_geometries(geoms,
                                              window_name="Objetos consolidados en el mapa")


if __name__ == "__main__":
    main()
