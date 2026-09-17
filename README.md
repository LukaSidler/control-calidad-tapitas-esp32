# Control de Calidad IA — Tapitas ESP32

Sistema de clasificación automática de tapitas de botella en una cinta
transportadora a escala, en tiempo real, sobre un **ESP32-P4** con cámara
**IMX708**. Clasifica cada tapita por dos criterios:

- **Color** (HSV), calculado directamente en el firmware.
- **Estado sana/rota**, con un modelo TinyML (CNN cuantizada a int8) que
  corre on-device con TFLite Micro — sin depender de un servidor externo
  para la inferencia.

## Contexto

Trabajo Práctico de la carrera de Ingeniería Electrónica, UTN Facultad
Regional Villa María. Entrega: 12 de noviembre de 2026.

## Estructura del repo

```
control-calidad-tapitas-esp32/
├── firmware/imx708_snapshot/   # Firmware ESP-IDF (basado en el ejemplo imx708_snapshot)
│   ├── main/                   # Código de la app: captura, HSV, inferencia TinyML, MQTT
│   ├── CMakeLists.txt
│   ├── sdkconfig.defaults      # Incluye el fix del bug de esp-nn (ver NOTAS_TINYML.md)
│   └── NOTAS_TINYML.md         # Detalle de la integración del modelo y el bug de esp-nn
├── modelo/                     # Entrenamiento, conversión y evaluación del modelo TinyML
├── dataset/                    # README + fotos de ejemplo (el dataset completo vive fuera del repo)
└── raspberry/                  # Ingesta MQTT + base de datos (pendiente)
```

## Compilar y flashear el firmware

Requiere **ESP-IDF v5.4** o superior.

```sh
cd firmware/imx708_snapshot
idf.py set-target esp32p4
idf.py build flash monitor
```

## Modelo TinyML: métricas

Evaluado sobre el test set (44 imágenes: 21 sana / 23 rota):

| Umbral | Accuracy | Precisión | Recall |
|---|---|---|---|
| 0.50 (default) | 86.4% | 100% | 73.9% |
| **0.20 (recomendado)** | 90.9% | 88.0% | **95.7%** |

Con el umbral por defecto (0.5) el modelo nunca marca una tapita sana como
rota (precisión perfecta), pero se le escapan ~26% de las tapitas rotas. En
control de calidad esto es al revés de lo deseable: un **falso negativo**
(tapa rota que pasa como sana) es mucho peor que un falso positivo (tapa
sana que se descarta de más). Por eso el sistema usa el umbral **0.20**,
que baja el recall perdido a un solo caso (95.7% de recall) a cambio de un
poco más de falsos positivos (88% de precisión). Detalle completo en
[`modelo/threshold_results.txt`](modelo/threshold_results.txt) y
[`modelo/README.md`](modelo/README.md).

El firmware manda `prob_rota` crudo (sin umbralizar) por MQTT, para poder
ajustar este umbral del lado de la Raspberry Pi sin reflashear el ESP32.

## Bug de esp-nn en ESP32-P4

Los kernels "optimizados" de `esp-nn` para ESP32-P4 dan resultados de
inferencia **incorrectos** para este modelo (probabilidad casi constante e
inútil, sin importar la tapita real). El fix es forzar los kernels ANSI-C
vía `sdkconfig.defaults` — ya está commiteado en este repo, así que
cualquiera que clone y compile desde cero no debería pisar el bug. Detalle
completo del diagnóstico en
[`firmware/imx708_snapshot/NOTAS_TINYML.md`](firmware/imx708_snapshot/NOTAS_TINYML.md).

## Estado actual

- ✅ Clasificación de color (HSV) y estado (sana/rota) funcionando en el
  ESP32-P4, validadas en banco de pruebas.
- ⏳ Integración con Raspberry Pi (ingesta MQTT + base de datos) pendiente
  — ver [`raspberry/README.md`](raspberry/README.md).

## Créditos

El firmware de `firmware/imx708_snapshot/` está basado en el ejemplo
`imx708_snapshot` del repositorio
[`esp32-p4-imx-camera`](https://github.com/mushbraindave/esp32-p4-imx-camera)
(Espressif / mushbraindave), modificado para agregar la clasificación por
color HSV y la inferencia TinyML sana/rota.

## Licencia

Apache License 2.0 (ver [`LICENSE`](LICENSE)), heredada del repositorio
base del que deriva el firmware.
