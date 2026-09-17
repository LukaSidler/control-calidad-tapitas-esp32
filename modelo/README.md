# Modelo de clasificación sana/rota para ESP32-P4 (TFLite Micro)

## Resultado final (test set, 44 imágenes: 21 sana / 23 rota)

| | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Modelo float (Keras) | 86.4% | 100% | 73.9% | 0.85 |
| Modelo int8 (.tflite) | 86.4% | 100% | 73.9% | 0.85 |

La cuantización a int8 **no degradó nada** — mismas predicciones exactas que el modelo float.

- **Precision 100%**: el modelo nunca marca una tapita sana como rota (cero falsos positivos).
- **Recall 73.9%**: se le escapan 6 de 23 tapitas rotas (falsos negativos). Revisé esas 6 fotos
  (`false_negatives.jpg`) y todas tienen **daño sutil**: un agujero chico, un desgarro en el borde,
  o grietas de estrés sin perforación oscura — a diferencia del agujero grande y oscuro que domina
  el resto del dataset de `rota`. El modelo aprendió bien esa forma de daño dominante, pero necesita
  más ejemplos de daño sutil para generalizar mejor a esos casos.

## Tamaño del modelo

- **`tapita_model_int8.tflite`: 112.0 KB** (input/output int8, listo para TFLite Micro)
- `tapita_model_float.tflite`: 400.9 KB (referencia, no usar en el micro)
- `tapita_model.keras`: modelo entrenable completo (para reentrenar o seguir iterando)

## Arquitectura final

CNN chica entrenada desde cero (no transfer learning — con ~220 imágenes de train no lo justificaba
frente al costo de tamaño de MobileNetV2):

```
Input 128x128x1 (escala de grises — ver por qué abajo)
Conv2D(16,3x3) → ReLU → MaxPool   (128→64)
Conv2D(32,3x3) → ReLU → MaxPool   (64→32)
Conv2D(64,3x3) → ReLU → MaxPool   (32→16)
Conv2D(128,3x3) → ReLU → MaxPool  (16→8)
GlobalMaxPooling2D → Dropout(0.2) → Dense(32, ReLU) → Dense(1, sigmoid)
```
~101K parámetros, sin BatchNorm.

## Decisiones de diseño (y por qué se cambiaron durante el desarrollo)

El plan original (ver conversación) proponía BatchNorm + GlobalAveragePooling + color RGB.
Las primeras corridas de entrenamiento no aprendían nada útil (val accuracy peor que el azar).
Diagnóstico paso a paso:

1. **Sin BatchNorm**: con solo ~14 pasos por época, las estadísticas acumuladas de BN nunca
   convergían — el modelo funcionaba en modo entrenamiento pero fallaba en modo inferencia
   (verificado comparando predicciones `training=True` vs `training=False`).
2. **GlobalMaxPooling en vez de GlobalAveragePooling**: la tarea es "¿hay un daño localizado en
   algún lado de la tapita?", no una estadística global de la imagen — promediar diluye una mancha
   oscura chica contra mucha superficie sana. Cambiar a max pooling fue la mejora más grande.
3. **Escala de grises en vez de RGB**: como `sana` se separó en train/val por color dominante (para
   evitar fugas de identidad — ver el README del agrupamiento), train y val terminan con paletas de
   color disjuntas para esa clase. El modelo empezaba a usar el color como atajo (ajustaba bien en
   train pero colapsaba en val). Como el color no aporta nada a esta tarea (ambas clases cubren
   todos los colores, y el sistema HSV ya lo maneja aparte), sacarlo de raíz fue más efectivo que
   pelearlo con augmentation de color, y de paso achica el input.

Validado con una prueba de sobreajuste en un subconjunto de 20 imágenes (llegó a 100% — confirmó que
el pipeline de datos y el flujo de gradientes estaban bien antes de tocar la arquitectura).

## Augmentation (solo en train, recalculado en cada época)
- Recorte fijo 1120×1120px alrededor del centro detectado + jitter aleatorio ±70px (simula
  descentrado real de producción)
- Rotación completa 0-360°, flip horizontal y vertical
- Zoom ±5%, contraste ±15%, brillo ±20 (en escala 0-255)
- Conversión a escala de grises

## Entrenamiento
- 80 épocas, Adam lr=1e-3 con ReduceLROnPlateau, EarlyStopping (restore_best_weights) sobre val_loss
- Batch size 16, sin BatchNorm, sin transfer learning

## Cuantización
- TFLite full-integer int8 (input y output int8), 150 imágenes de train (con augmentation típico)
  como dataset representativo para calibración

## Integración en el ESP32-P4
- El modelo espera un recorte cuadrado de la tapita ya localizado (no el frame completo) — ver
  discusión de diseño en la conversación sobre el pipeline de recorte.
- Input: 128×128, 1 canal (escala de grises), int8 (escala/zero-point en el propio `.tflite`).
- Si el firmware recibe la imagen en color, convertir a escala de grises antes de cuantizar.

## Archivos
- `tapita_model_int8.tflite` — el modelo para deployar
- `tapita_model_float.tflite`, `tapita_model.keras` — referencia / reentrenamiento
- `train.py`, `convert_tflite.py`, `eval_tflite.py` — pipeline completo reproducible
- `geometry_final.json`, `split_plan_final2.json` — geometría de recorte y partición train/val/test
- `false_negatives.jpg` — las 6 tapitas rotas que el modelo no detectó, para inspección
