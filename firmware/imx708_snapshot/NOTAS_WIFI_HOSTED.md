# Notas: WiFi (ESP32-P4 + ESP32-C6 vía esp-hosted-mcu) — bug conocido, no resuelto

## Qué pasa

El Waveshare ESP32-P4-WIFI6 no tiene WiFi nativo: usa un ESP32-C6 como
coprocesador de red conectado por SDIO (4-bit, 40MHz), vía
[`esp-hosted-mcu`](https://github.com/espressif/esp-hosted-mcu). Bajo esta
arquitectura, tener WiFi asociado (`esp_wifi_connect`) **más** cualquier
otra tarea FreeRTOS que haga bit-banging de GPIO en loop activo (en
nuestro caso, el control del motor paso a paso de la cinta transportadora)
dispara, de forma extremadamente reproducible, un crash sin recuperación:

```
E sdmmc_io: sdmmc_io_rw_extended: sdmmc_send_cmd returned 0x107
E H_SDIO_DRV: sdio_write_task: 0: Failed to send data: 263 54 54
E H_SDIO_DRV: Unrecoverable host sdio state
```

Reportado y documentado extensamente en
[`espressif/esp-hosted-mcu#167`](https://github.com/espressif/esp-hosted-mcu/issues/167)
(issue abierto al momento de escribir esto, sin fix confirmado upstream).
Nuestra propia investigación está en ese hilo (comentario de
[`LukaSidler`, 11-sep-2026](https://github.com/espressif/esp-hosted-mcu/issues/167#issuecomment-5641327767)):
aislamos el fallo a un timing fijo (~56s después de asociar WiFi),
totalmente insensible a las características del motor (corriente,
velocidad, prioridad de la tarea, núcleo al que está fijada, yields por
`vTaskDelay`) — lo que apunta a una carrera de timing dentro del stack
WiFi/SDIO del host, no a integridad de señal de la placa.

## Workaround usado en este proyecto

Se sacó WiFi/MQTT del ESP32 por completo. El ESP32 nunca vuelve a tocar el
C6 para nada relacionado a red:

- La clasificación (color HSV + `prob_rota`) viaja por el mismo cable
  serial/USB que ya se usaba para transferir las imágenes (ver
  `publish_clasificacion_por_serial()` en `main/imx708_snapshot_main.c`).
- `ver_imx708.py`, corriendo en una PC, lee esa línea por serial y es
  quien la publica por MQTT hacia la Raspberry Pi.
- Por el mismo motivo, la imagen capturada tampoco se manda por MQTT: una
  transferencia de ~90KB+ por el enlace SDIO P4↔C6 sería justamente el
  tipo de tráfico que dispara el bug. La imagen sigue viajando por serial.

Esto evita cualquier uso de WiFi en el ESP32 durante la operación normal
(captura + control de motor), así que el bug nunca se llega a disparar.

## Si en algún momento se quiere usar WiFi directo desde el ESP32

Revisar el estado de
[`espressif/esp-hosted-mcu#167`](https://github.com/espressif/esp-hosted-mcu/issues/167)
antes de intentarlo — a la fecha de este commit sigue abierto y sin
solución confirmada. Si el fix llega upstream, se podría volver a publicar
MQTT directo desde el ESP32 y sacar el puente por PC.
