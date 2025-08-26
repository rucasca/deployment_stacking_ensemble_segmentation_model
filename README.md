
# DEPLOYMENT

Sistema web para la productivización de diferentes modelos de segmentación semántica mediante el framework Dash.

Los modelos productivizados son los siguientes:
1. Baseline U-Net.
2. Ensemble conformado por Yolov8 y Segment Anything Model.
3. Ensemble conformado por Retinanet y Segment Anything Model.
4. Ensemble conformado por Segment Anything Model y CLIP.
5. Ensemble tipo stacking final conformado por Retinanet, Segment Anything Model y U-Net.

Para su uso, se han de seguir los siguientes pasos:
1. Generación de un entorno virtual de poetry haciendo uso de las librerias almacenadas en el fichero `pyproject.toml`
2. Descarga de los modelos a emplear de las siguientes direcciones:
    - SAM:
    - YOLOv8
    - CLIP
    - UNET_Baseline
    - UNET_Final
3. Edición de las variables del entorno del fichero `.env` con las direcciones del PC donde se han descargado los archivos.
4. Ejecución del fichero `app.py` y búsqueda en el navegador



