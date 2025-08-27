
# 06_DEPLOYMENT

Sistema web implementado usando el framework de Python **Dash** para la productivización de diferentes modelos de segmentación semántica.

## 📌 Modelos productivizados
1. **Baseline U-Net**
2. **Ensemble conformado por Yolov8 y Segment Anything Model**
3. **Ensemble conformado por Retinanet y Segment Anything Model**
4. **Ensemble conformado por Segment Anything Model y CLIP**
5. **Ensemble tipo stacking final conformado por Retinanet, Segment Anything Model y U-Net**

## ⚙️ Pasos para la ejecución
Para su uso, se han de seguir los siguientes pasos:
1. Generación de un entorno virtual de `Poetry` para poder hacer uso de las librerias y sus respectivas versiones almacenadas en el fichero `pyproject.toml`. Para ello, en primera instancia se descarga poetry en nuestro entorno mediante *poetry install* y posteriormente ejecutamos *poetry install*, que descargará las librerias. Una vez hecho esto, se selecciona el entorno creado como entorno de ejecución mediante *poetry shell*.
2. Descarga de los modelos a emplear que requieran descargas manuales en las siguientes direcciones:
    - SAM: [`https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints`](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints)
    - UNET_Baseline:[`https://github.com/rucasca/TFM/tree/master/models/trained_models`](https://github.com/rucasca/TFM/tree/master/models/trained_models)
    - UNET_Final: [`https://github.com/rucasca/TFM/tree/master/models/trained_models`](https://github.com/rucasca/TFM/tree/master/models/trained_models)
3. **Edición de las variables del entorno** del fichero `.env` con las direcciones del PC donde se han descargado los archivos.
4. Ejecución del servicio web mediante el comando *python app.py*
5. **Acceso al navegador** en la dirección.



