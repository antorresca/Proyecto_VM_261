"""
Validación de dimensiones — Módulo P1 (Fase 6)
----------------------------------------------
Compara las dimensiones consolidadas del pipeline (objetos_consolidados.csv,
salida de consolidar_detecciones.py) contra medidas tomadas con cinta métrica
en el lab, y reporta el % de error por dimensión y por objeto.

Recordatorio de encuadre: las dims del pipeline son una ESTIMACIÓN derivada
del mapeo 3D (~10-15% de error esperado), no metrología. Esta validación
sirve para confirmar que el error queda dentro de ese rango, no para
certificar precisión.

Cómo se comparan los ejes: la caja OBB reporta d90x/d90y en el marco propio
de la caja (depende del yaw), así que "x" y "y" no corresponden fijo a
"ancho" y "fondo". Para no depender de la orientación, se compara:
  - alto        = d90z            vs  alto medido
  - lado mayor  = max(d90x, d90y) vs  max(ancho, fondo) medidos
  - lado menor  = min(d90x, d90y) vs  min(ancho, fondo) medidos

Uso (2 pasos):
  1. Generar la plantilla para llenar con la cinta métrica (P2):
       python validar_dims.py --plantilla
       python validar_dims.py --plantilla --clases silla mesa estante
     -> crea medidas_reales.csv con los objetos 'ok' del consolidado.
     Llenar alto_cm / ancho_cm / fondo_cm EN CENTIMETROS (acepta coma o
     punto decimal). Dejar en blanco los objetos que no se midan.

  2. Comparar:
       python validar_dims.py
       python validar_dims.py --dim mediana     # usar dims mediana en vez de p90
     -> imprime la tabla y guarda validacion_dims.csv
"""

import os
import csv
import argparse
import numpy as np

DIR = os.path.join(os.path.expanduser("~"), "Proyecto_ZED_P1")
RANGO_ESPERADO = (10.0, 15.0)   # % de error esperado del mapeo


def num(texto):
    """Float tolerante: acepta '85', '85.5', '85,5'. ''/None/0 -> None (no medido)."""
    if texto is None:
        return None
    t = str(texto).strip().replace(",", ".")
    if not t:
        return None
    try:
        v = float(t)
    except ValueError:
        return None
    return v if v > 0 else None      # 0 = "no lo medí", no una medida real


def abrir_medidas(ruta):
    """DictReader tolerante con Excel: detecta separador (, ; tab) y BOM."""
    f = open(ruta, newline="", encoding="utf-8-sig")
    muestra = f.read(4096)
    f.seek(0)
    delim = max([",", ";", "\t"], key=muestra.count)
    return csv.DictReader(f, delimiter=delim)


def cargar_consolidado(ruta):
    objs = {}
    with open(ruta, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            objs[int(r["id"])] = r
    return objs


def hacer_plantilla(objs, ruta, clases, incluir_dudosos):
    filas = []
    for oid, r in sorted(objs.items(), key=lambda kv: -int(kv[1]["frames_visto"])):
        if r["estado"] == "alto_luz":
            continue
        if r["estado"] == "dudoso" and not incluir_dudosos:
            continue
        if clases and r["clase"] not in clases:
            continue
        filas.append([oid, r["clase"], r["frames_visto"], "", "", "", ""])
    if not filas:
        print("[ERROR] Ningún objeto cumple los filtros para la plantilla.")
        return
    if os.path.exists(ruta):
        print(f"[ERROR] {ruta} ya existe — no lo sobreescribo para no perder "
              f"medidas ya tomadas. Bórralo o muévelo si quieres regenerarlo.")
        return
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "clase", "frames_visto", "alto_cm", "ancho_cm",
                    "fondo_cm", "nota"])
        w.writerows(filas)
    print(f"[OK] Plantilla con {len(filas)} objetos -> {ruta}")
    print("     Llenar alto_cm/ancho_cm/fondo_cm EN CM con la cinta métrica.")
    print("     Identifica cada objeto por su id/clase en vista_arriba.png.")
    print("     Con 2-3 objetos bien medidos basta para la Fase 6; deja en")
    print("     blanco los demás.")


def comparar(objs, ruta_medidas, dim, ruta_out):
    if not os.path.exists(ruta_medidas):
        print(f"[ERROR] No existe {ruta_medidas}. Genera la plantilla primero:\n"
              f"  python validar_dims.py --plantilla")
        return
    pref = {"p90": ("d90x", "d90y", "d90z"), "mediana": ("dmx", "dmy", "dmz")}[dim]

    filas_out = []
    errores_todos = []
    print(f"\n===== VALIDACION dims {dim} vs cinta métrica =====")
    print("(estimación derivada del mapeo; error esperado ~10-15%)\n")

    rd = abrir_medidas(ruta_medidas)
    if not rd.fieldnames or "alto_cm" not in [c.strip() for c in rd.fieldnames]:
        print(f"[ERROR] No encuentro la columna alto_cm en {ruta_medidas} "
              f"(columnas: {rd.fieldnames}). ¿Excel cambió los encabezados?")
        return
    if True:
        for r in rd:
            alto = num(r.get("alto_cm"))
            ancho = num(r.get("ancho_cm"))
            fondo = num(r.get("fondo_cm"))
            if alto is None and ancho is None and fondo is None:
                continue                      # objeto sin medir
            oid = int(r["id"])
            if oid not in objs:
                print(f"[WARN] id {oid} no está en el consolidado — lo salto "
                      f"(¿corrida distinta? revisa que el CSV consolidado sea "
                      f"el de la misma grabación).")
                continue
            o = objs[oid]
            if r["clase"] != o["clase"]:
                print(f"[WARN] id {oid}: clase '{r['clase']}' en medidas vs "
                      f"'{o['clase']}' en consolidado — verifica el id.")

            est = [float(o[pref[0]]) * 100, float(o[pref[1]]) * 100,
                   float(o[pref[2]]) * 100]           # a cm
            # emparejar sin depender del yaw de la caja
            pares = [("alto", est[2], alto)]
            horiz_est = sorted(est[:2], reverse=True)
            medidos = [m for m in (ancho, fondo) if m is not None]
            if len(medidos) == 2:
                horiz_real = sorted(medidos, reverse=True)
                pares += [("lado mayor", horiz_est[0], horiz_real[0]),
                          ("lado menor", horiz_est[1], horiz_real[1])]
            elif len(medidos) == 1:
                # solo un lado medido: compararlo contra el est más cercano
                e = min(horiz_est, key=lambda v: abs(v - medidos[0]))
                pares += [("lado", e, medidos[0])]

            errs_obj = []
            print(f"#{oid} {o['clase']}  ({o['frames_visto']} vistas, "
                  f"estado {o['estado']})")
            for nombre, e, m in pares:
                if m is None:
                    continue
                if m <= 0:
                    print(f"    {nombre:<10} medida inválida ({m}) — salto")
                    continue
                err = (e - m) / m * 100
                errs_obj.append(abs(err))
                filas_out.append([oid, o["clase"], nombre, round(e, 1),
                                  round(m, 1), round(e - m, 1), round(err, 1)])
                print(f"    {nombre:<10} est {e:6.1f} cm | real {m:6.1f} cm | "
                      f"error {err:+6.1f}%")
            if errs_obj:
                mae = np.mean(errs_obj)
                errores_todos += errs_obj
                v = ("DENTRO del rango esperado" if mae <= RANGO_ESPERADO[1]
                     else "FUERA del rango esperado")
                print(f"    -> error medio |{mae:.1f}%|  [{v} ~{RANGO_ESPERADO[0]:.0f}"
                      f"-{RANGO_ESPERADO[1]:.0f}%]")
            print()

    if not filas_out:
        print("[ERROR] La plantilla no tiene medidas llenas todavía "
              "(alto_cm/ancho_cm/fondo_cm vacíos).")
        return

    mae_g = np.mean(errores_todos)
    peor = max(errores_todos)
    print("===== RESUMEN =====")
    print(f"  {len(filas_out)} dimensiones comparadas en "
          f"{len(set(f[0] for f in filas_out))} objetos")
    print(f"  Error absoluto medio: {mae_g:.1f}%   | peor dimensión: {peor:.1f}%")
    if mae_g <= RANGO_ESPERADO[1]:
        print(f"  VEREDICTO: dentro del ~{RANGO_ESPERADO[0]:.0f}-"
              f"{RANGO_ESPERADO[1]:.0f}% esperado para estimación de mapeo. "
              f"Fase 6 OK.")
    else:
        print(f"  VEREDICTO: por encima del ~{RANGO_ESPERADO[1]:.0f}% esperado. "
              f"Revisa: ¿grabación baja (caras no vistas)?, ¿--al-piso aplicado?, "
              f"¿objeto con pocas vistas u ocluido?")

    with open(ruta_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "clase", "dimension", "estimado_cm", "real_cm",
                    "dif_cm", "error_pct"])
        w.writerows(filas_out)
        w.writerow([])
        w.writerow(["RESUMEN", "", "", "", "",
                    "error_abs_medio_pct", round(mae_g, 1)])
    print(f"\n[OK] Tabla -> {ruta_out}")


def main():
    ap = argparse.ArgumentParser(description="Validar dims consolidadas vs cinta métrica")
    ap.add_argument("--csv", default=os.path.join(DIR, "objetos_consolidados.csv"),
                    help="CSV consolidado (salida de consolidar_detecciones.py)")
    ap.add_argument("--medidas", default=os.path.join(DIR, "medidas_reales.csv"),
                    help="CSV de medidas con cinta métrica (en cm)")
    ap.add_argument("--out", default=os.path.join(DIR, "validacion_dims.csv"))
    ap.add_argument("--dim", choices=["p90", "mediana"], default="p90",
                    help="Qué dims del consolidado usar (default p90)")
    ap.add_argument("--plantilla", action="store_true",
                    help="Generar medidas_reales.csv para llenar en el lab")
    ap.add_argument("--clases", nargs="*", default=None,
                    help="En --plantilla: limitar a estas clases "
                         "(p.ej. --clases silla mesa estante)")
    ap.add_argument("--incluir-dudosos", action="store_true",
                    help="En --plantilla: incluir también objetos 'dudoso'")
    args = ap.parse_args()

    objs = cargar_consolidado(args.csv)
    if not objs:
        print(f"[ERROR] {args.csv} vacío o ilegible."); return
    print(f"[OK] {len(objs)} objetos en {args.csv}")

    if args.plantilla:
        hacer_plantilla(objs, args.medidas, args.clases, args.incluir_dudosos)
    else:
        comparar(objs, args.medidas, args.dim, args.out)


if __name__ == "__main__":
    main()
