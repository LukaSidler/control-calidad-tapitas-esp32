# Raspberry Pi — ingesta y base de datos

Pendiente.

El ESP32 manda `prob_rota` (float crudo 0–1) por el puente serial hacia
una PC (`ver_imx708.py`), que es quien lo publica por MQTT — el ESP32 no
usa WiFi/MQTT directo por un bug conocido de `esp-hosted-mcu` (ver
[`NOTAS_WIFI_HOSTED.md`](../firmware/imx708_snapshot/NOTAS_WIFI_HOSTED.md)
en el README principal). Acá va a vivir `tapitas_ingest.py`, que va a:

- Suscribirse al tópico MQTT donde el ESP32 publica cada clasificación
  (color HSV + `prob_rota`).
- Aplicar el umbral de decisión (**0.20**, ver
  [`modelo/threshold_results.txt`](../modelo/threshold_results.txt)) para
  convertir `prob_rota` en sana/rota.
- Cargar cada resultado a una base de datos.

Broker MQTT: Raspberry Pi 3B+ en la IP local de la Raspberry (configurable vía `TAPITAS_MQTT_HOST` en `ver_imx708.py`).
