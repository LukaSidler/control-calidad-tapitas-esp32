# Notas: integración del modelo TinyML (sana/rota) en el firmware

## Qué hace

`tinyml_estado.cc` corre el modelo `tapita_model_int8.tflite` (CNN entrenada
offline sobre fotos de tapitas, ver `dataset_tapitas/modelo/`) directamente
en el ESP32-P4 usando TFLite Micro (`esp-tflite-micro`), sobre cada frame
capturado por la IMX708. La salida es `prob_rota`: probabilidad (0.0–1.0) de
que la tapita esté rota, que se manda por MQTT junto con el color HSV.

Preprocesamiento (recorte 1120×1120 centrado, resize a 128×128, escala de
grises, cuantización a int8) reproduce **exactamente** el pipeline de
entrenamiento (`train.py` / `threshold_analysis.py`) — está documentado en
detalle en los comentarios de `tinyml_estado.cc`.

## ⚠️ Bug crítico ya resuelto — no revertir esta config

El `sdkconfig` de este proyecto (`imx708_snapshot/sdkconfig`, sección
`ESP-NN`) tiene que tener:

```
CONFIG_NN_ANSI_C=y
# CONFIG_NN_OPTIMIZED is not set
CONFIG_NN_OPTIMIZATIONS=0
```

Se configura en `idf.py menuconfig` → `ESP-NN` → elegir kernels **ANSI-C**
en vez de **optimizados** (PIE/SIMD).

### Por qué

Los kernels "optimizados" de `esp-nn` para ESP32-P4 (el default de la
librería) dan resultados numéricamente **incorrectos** para este modelo en
particular. Se detectó porque con el default (`CONFIG_NN_OPTIMIZED=y`) el
modelo daba `prob_rota` siempre alto (~0.96+) sin importar si la tapita
mostrada era sana o rota — un output prácticamente constante.

Se diagnosticó comparando, para la misma foto capturada, el tensor de
entrada ya cuantizado (idéntico entre el firmware en C++ y una réplica en
Python/`tflite_runtime` del mismo preprocesamiento) contra la salida del
modelo: con el input casi idéntico, Python daba `prob≈0.14` (sana,
correcto) y el ESP32 con kernels optimizados daba `prob≈0.97` (rota,
incorrecto) — la única diferencia posible era la implementación de los
kernels de inferencia, no el preprocesamiento ni la cuantización.

`esp-nn` expone justamente esta opción (`Kconfig.projbuild` del propio
componente) documentando los kernels ANSI-C como los de referencia "para
verificación y debug". Cambiando a ANSI-C el bug desapareció: la misma
tapita sana pasó a dar `prob_rota≈0.17`, y una tapita rota real dio
`prob_rota≈0.996` — separación clara y correcta en ambos sentidos
(validado el 17-sep-2026).

### Síntoma si esto se resetea

Si algún día un `idf.py menuconfig`, un `idf.py fullclean`, o una
actualización de la dependencia `esp-nn` vuelve a dejar
`CONFIG_NN_OPTIMIZED=y`, el síntoma va a ser el mismo de antes:
**`prob_rota` pegado cerca de un extremo (típicamente ~0.9+) sin importar
la tapita real que se muestre a la cámara.** Si eso pasa, revisar primero
esta sección del `sdkconfig` antes de sospechar del modelo, del dataset o
del preprocesamiento — todo eso ya estaba probado y funcionando bien.

### Costo del fix

Los kernels ANSI-C son más lentos que los optimizados (sin SIMD/PIE), pero
en las pruebas el tiempo de inferencia (`invoke`) se mantuvo en el orden de
1–2 segundos por frame, que es aceptable para este sistema (clasificación
por tapita en una cinta a escala, no tiempo real de alta frecuencia). Si en
el futuro el tiempo de ciclo se vuelve un problema, se podría investigar
si el bug es específico de alguna combinación de ops/shapes de este modelo
en particular (para eventualmente usar kernels optimizados salvo en la
operación problemática), pero no es necesario para este proyecto.

## Umbral de decisión sana/rota

El análisis offline sobre el test set (`threshold_analysis.py`,
`threshold_results.txt`) recomienda un umbral de **0.20** (en vez del 0.5
por defecto) para priorizar recall — en control de calidad, un falso
negativo (tapita rota que pasa como sana) es peor que un falso positivo.
Con el modelo funcionando bien on-device, esta recomendación offline sigue
siendo válida (no dependía de la inferencia en el ESP32).

El firmware manda `prob_rota` crudo por MQTT (no un booleano ya decidido),
para poder ajustar el umbral desde el lado de ingesta (Raspberry Pi,
`tapitas_ingest.py`) sin tener que reflashear el ESP32.
