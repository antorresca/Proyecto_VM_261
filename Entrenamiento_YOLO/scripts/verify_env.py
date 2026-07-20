from __future__ import annotations

import platform


def main() -> None:
    print(f"Python: {platform.python_version()}")

    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"VRAM: {props.total_memory / 1024**3:.2f} GB")
    except Exception as exc:
        print(f"PyTorch no disponible: {exc}")

    for module_name, label in [
        ("ultralytics", "Ultralytics"),
        ("cv2", "OpenCV"),
        ("open3d", "Open3D"),
    ]:
        try:
            module = __import__(module_name)
            print(f"{label}: {getattr(module, '__version__', 'ok')}")
        except Exception as exc:
            print(f"{label} no disponible: {exc}")

    try:
        import pyzed.sl as sl  # type: ignore
        print(f"ZED SDK / pyzed: disponible ({sl.Camera().get_sdk_version()})")
    except Exception as exc:
        print(f"ZED SDK / pyzed: no disponible ({exc})")


if __name__ == "__main__":
    main()
