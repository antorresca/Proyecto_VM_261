from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrae frames RGB desde archivos SVO/SVO2 de ZED.")
    parser.add_argument("--svo", required=True, help="Ruta al archivo .svo o .svo2.")
    parser.add_argument("--out", default="data/labelme", help="Carpeta de salida.")
    parser.add_argument("--every", type=int, default=30, help="Guardar 1 frame cada N frames.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 significa sin limite.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    svo_path = Path(args.svo)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not svo_path.exists():
        raise FileNotFoundError(f"No existe el SVO/SVO2: {svo_path}")

    import pyzed.sl as sl  # type: ignore

    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.svo_real_time_mode = False
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"No se pudo abrir {svo_path}: {status}")

    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    saved = 0
    frame_idx = 0
    try:
        while True:
            err = zed.grab(runtime)
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if err != sl.ERROR_CODE.SUCCESS:
                frame_idx += 1
                continue
            if frame_idx % args.every == 0:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                rgba = image.get_data()
                bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                out = out_dir / f"{svo_path.stem}_frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out), bgr)
                saved += 1
                if args.max_frames and saved >= args.max_frames:
                    break
            frame_idx += 1
    finally:
        zed.close()
    print(f"Frames guardados: {saved}")
    print(f"Salida: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
