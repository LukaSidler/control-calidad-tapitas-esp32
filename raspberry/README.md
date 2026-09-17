# Raspberry Pi — ingesta y base de datos

Pendiente.

El ESP32 manda `prob_rota` (float crudo 0–1) por MQTT. Acá va a vivir
`tapitas_ingest.py`, que va a:

- Suscribirse al tópico MQTT donde el ESP32 publica cada clasificación
  (color HSV + `prob_rota`).
- Aplicar el umbral de decisión (**0.20**, ver
  [`modelo/threshold_results.txt`](../modelo/threshold_results.txt)) para
  convertir `prob_rota` en sana/rota.
- Cargar cada resultado a una base de datos.

Broker MQTT: Raspberry Pi 3B+ en `192.168.1.100`.
