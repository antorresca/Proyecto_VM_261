# Modelo YOLO11-seg para robot

Modelos de la corrida final de segmentación de objetos con imágenes ZED.

## Archivos incluidos

- `best.pt`: modelo PyTorch/Ultralytics usado por el pipeline 3D.
- `best.onnx`: exportación portable para ONNX Runtime, C++ o ROS.
- `classes.txt`: nueve clases en el orden del entrenamiento.
- `dataset.yaml`: metadatos portables del dataset.

La entrada esperada es una imagen de 640 × 640 píxeles. El modelo devuelve cajas, clases, confianza y máscaras de segmentación.

Los formatos TorchScript, OpenVINO o TensorRT pueden regenerarse mediante `../scripts/export_model.py`. TensorRT debe generarse en el equipo de destino porque el archivo `.engine` depende de la GPU, CUDA, TensorRT y los controladores instalados.
