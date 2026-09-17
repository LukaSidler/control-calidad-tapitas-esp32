# Dataset

Las imágenes crudas usadas para entrenar el modelo **no están versionadas en
este repo** (162 fotos de tapitas sanas + 153 de tapitas rotas, ~80 MB en
total — no tiene sentido cargar eso a git).

Viven localmente en:

```
dataset_tapitas/sana/   (162 imágenes, sana_0001.jpg ... sana_0162.jpg)
dataset_tapitas/rota/   (153 imágenes, rota_0001.jpg ... rota_0153.jpg)
```

fuera de este repositorio, como hermanas de la carpeta `dataset_tapitas/modelo/`
(cuyo contenido sí está en `../modelo/`).

`modelo/train.py` asume esa estructura de carpetas (`dataset_tapitas/sana/` y
`dataset_tapitas/rota/`) para reentrenar el modelo desde cero. Para
reproducir el entrenamiento, recreá esas dos carpetas con las fotos
correspondientes al lado de `modelo/`, o ajustá las rutas en `train.py`.

## Ejemplos

En `ejemplos/` hay 3 fotos de cada clase a modo ilustrativo, para tener una
idea del tipo de imagen sin necesitar el dataset completo.
