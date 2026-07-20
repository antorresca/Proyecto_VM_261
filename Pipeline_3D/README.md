# Módulo P1 — Geometría 3D y Mapeo (cámara ZED2)

Código del módulo de **geometría 3D / mapeo** del proyecto de detección de objetos en 3D con cámara estéreo ZED2. Este módulo toma una grabación `.svo/.svo2` de la ZED2 y el detector 2D YOLO entrenado por el equipo (`best.pt`) y produce:

1. Un **mapa 3D** del entorno (nube de puntos fusionada del recorrido).
2. **Cajas 3D orientadas (OBB)** por objeto detectado: clase, posición, dimensiones y orientación en coordenadas del mapa.
3. Una **consolidación multivista**: agrupa las detecciones de todos los frames por objeto físico.

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── replay_svo.py               # Prueba de humo: RGB + profundidad + nube de 1 frame
├── mapeo_manual.py             # Paso 1: mapa 3D del recorrido (tracking + fusión propia)
├── limpiar_mapa.py             # Paso 2: limpieza del mapa (flotantes de ventanas/luz)
├── deteccion_obb.py            # Paso 3: YOLO 2D -> recorte de nube -> OBB 3D por frame
├── limpieza_obb.py             # Librería usada por deteccion_obb (limpieza + OBB al plano)
├── vista_lado.py               # Paso 4 (diagnóstico): alzado del mapa + detecciones
├── consolidar_detecciones.py   # Paso 5: agrupar detecciones por objeto físico
├── video_demo.py               # Extra: MP4 demo (video etiquetado + mapa 3D con cajas)
└── ejemplos/                   # Salidas reales de referencia (formato de los CSV/PNG)
    ├── detecciones_3d.csv
    ├── objetos_consolidados.csv
    └── vista_arriba.png
```

Todos los scripts se ejecutan **desde esta carpeta** (usan rutas relativas entre sí: `deteccion_obb.py` importa `limpieza_obb.py`).

**Salidas:** los scripts escriben sus resultados en `~/Proyecto_ZED_P1/` (se crea sola en el home del usuario, p. ej. `C:\Users\<tu_usuario>\Proyecto_ZED_P1\`). Ahí quedan `mapa_manual.ply`, `mapa_limpio.ply`, `detecciones_3d.csv`, `objetos_consolidados.csv`, `vista_arriba.png`, etc.

## Requisitos

- **Windows 10/11** con **GPU NVIDIA** (el ZED SDK requiere CUDA).
- **ZED SDK 5.x** (probado con 5.3.0) — incluye CUDA y la API de Python `pyzed`.
- **Python entre 3.10 y 3.12**.
- Archivos que **no** están en el repo (por tamaño):
  - una grabación `.svo` / `.svo2` hecha con la ZED2 ;
  - el modelo entrenado `best.pt`.

## Instalación

1. **ZED SDK**: descargar e instalar desde [stereolabs.com/developers](https://www.stereolabs.com/developers/release/) (marca la opción de instalar CUDA si no lo tienes).

2. **Entorno de Python** (desde esta carpeta):

   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **pyzed** (API de Python del ZED SDK) — con el venv activado:

   ```bat
   python "C:\Program Files (x86)\ZED SDK\get_python_api.py"
   ```

4. **Verificar** con la prueba de humo (deben abrirse RGB, profundidad y nube 3D):

   ```bat
   python replay_svo.py "ruta\a\tu_grabacion.svo2"
   ```

5. Copiar `best.pt` a esta carpeta (o pasar su ruta con `--modelo`).

## Uso — pipeline completo

Con una grabación a 30 fps se procesa 1 de cada 3 frames (`--cada 3`); si la grabación es a 60 fps, usa `--cada 6`.

```bat
:: 1. Mapa 3D del recorrido
python mapeo_manual.py "ruta\a\grabacion.svo2" --depth neural_plus

:: 2. Limpiar el mapa (quita puntos flotantes de ventanas/luz)
python limpiar_mapa.py --brillo 0.97 --sin-visor

:: 3. Detección 3D frame a frame (YOLO -> recorte -> OBB), en coords. del MAPA
python deteccion_obb.py "ruta\a\grabacion.svo2" --modelo best.pt --cada 3 --mundo --depth neural_plus

:: 4. Diagnóstico: ¿el piso quedó en z=0? ¿mapa y CSV son de la misma corrida?
python vista_lado.py mapa_limpio.ply detecciones_3d.csv

:: 5. Consolidar: de ~2000 detecciones a la lista de objetos físicos
python consolidar_detecciones.py --sin-clases ventana --z-max 2.0 --radio 0.6 --min-det 5 --al-piso silla mesa estante --ver mapa_limpio.ply

Cada script acepta `--help` con la descripción completa de sus opciones.

### Resultado de referencia

Sobre una grabación del laboratorio (`--cada 3 --mundo`): **2 164 detecciones por frame → 37 objetos físicos consolidados** (10 sillas, 18 mesas, 3 estantes, 2 computadores, 3 tableros, 1 puerta). En `ejemplos/` están los CSV y el plano cenital (`vista_arriba.png`) de esa corrida, para ver el formato de salida sin ejecutar nada.

## Decisiones de diseño y convenciones

- **Sistema de coordenadas:** `RIGHT_HANDED_Z_UP`, en metros, para todo el equipo.
- **Origen del mundo = PISO** (`set_floor_as_origin=True` en `mapeo_manual.py` y `deteccion_obb.py`): el SDK detecta el plano del suelo y pone `z=0` ahí, sin importar a qué altura arrancó la cámara. Requisito: el suelo debe ser visible en los primeros frames de la grabación. El flag `--origen-camara` recupera el comportamiento clásico (origen donde arrancó la cámara).
- **Mapeo manual propio en vez de Spatial Mapping del SDK:** el módulo Spatial Mapping crasheaba en el hardware disponible, así que el mapa se construye acumulando la nube de cada frame transformada con la pose del positional tracking, con voxel-downsample periódico (`mapeo_manual.py`).
- **2D→3D con pipeline propio en vez del "custom box objects" del SDK:** cada caja 2D de YOLO recorta la nube organizada del frame; el recorte se limpia (voxel, outliers, RANSAC del plano de soporte, DBSCAN) y se le ajusta una OBB anclada al plano (`limpieza_obb.py`).
- **Supresión de solape por clase** en `deteccion_obb.py`: umbral estricto entre cajas de la misma clase (`--solape 0.05`, duplicados de YOLO) y tolerante entre clases distintas (`--solape-cruzado 0.60`), para que una silla frente a una mesa no elimine la detección de menor confianza.
- **Consolidación multivista:** cada frame ve solo una cara del objeto, así que la caja por frame queda corta; el percentil 90 de las dimensiones sobre todas las vistas se acerca al tamaño real. Las clases apoyadas en el suelo pueden extruirse hasta `z=0` (`--al-piso`).

## Notas y problemas conocidos

- En modo `--mundo` **no se saltan frames al leer el SVO** (el seek rompe el tracking); `--cada N` solo decide en qué frames corre YOLO.
- El mensaje `END OF SVO FILE REACHED` al final de una corrida es normal (fin de la grabación).
- `deteccion_obb.py` **sobrescribe** `detecciones_3d.csv` en cada corrida; renombra los resultados que quieras conservar.
- `limpiar_mapa.py --brillo` puede borrar superficies blancas reales (p. ej. un techo blanco); si el techo desaparece del mapa, baja o quita ese filtro.
- En sesiones de escritorio remoto los visores 3D (OpenGL) pueden fallar; el pipeline y los PNG se generan igual (`--sin-visor`, sin `--ver`, `--guardar-jpg`).
- Si las cajas "flotan" sobre el piso, corre `vista_lado.py`: detecta si el origen del mundo no quedó en el piso e imprime los flags corregidos.

