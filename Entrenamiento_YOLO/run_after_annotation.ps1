$ErrorActionPreference = "Stop"

.\.venv\Scripts\python.exe scripts\labelme_to_yolo_seg.py `
  --labelme-dir data\labelme `
  --images-out data\images\raw_annotated `
  --labels-out data\labels\raw

.\.venv\Scripts\python.exe scripts\split_dataset.py `
  --images data\images\raw_annotated `
  --labels data\labels\raw `
  --val-ratio 0.2

.\.venv\Scripts\python.exe scripts\update_dataset_yaml.py

.\.venv\Scripts\python.exe scripts\train.py `
  --model yolo11n-seg.pt `
  --data configs\dataset.yaml `
  --epochs 80 `
  --imgsz 640 `
  --batch 8 `
  --name zed_real_yolo11n
