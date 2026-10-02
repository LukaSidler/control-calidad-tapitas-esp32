# Raspberry Pi — ingesta, base de datos y dashboard web

El ESP32 no publica por MQTT directo (bug de `esp-hosted-mcu`, ver
[`NOTAS_WIFI_HOSTED.md`](../firmware/imx708_snapshot/NOTAS_WIFI_HOSTED.md)):
la clasificación viaja por serial hasta la PC, donde
[`ver_imx708.py`](../firmware/imx708_snapshot/ver_imx708.py) la publica en
el broker Mosquitto de la Raspberry, junto con la foto y los resultados de
la IA (Gemini / Ollama).

| Archivo | Qué hace |
|---|---|
| `tapitas_ingest.py` | Suscriptor MQTT → SQLite (`~/tapitas/clasificacion.db`) + fotos en `~/tapitas/imagenes/`. Aplica el umbral sana/rota/dudoso sobre `prob_rota`. |
| `tapitas_dashboard.py` | Panel web (FastAPI) en el puerto 8000: API REST + WebSocket en vivo. Solo lee la DB (salvo "Borrar historial"). |
| `dashboard.html` | La página del panel (un solo archivo, Chart.js por CDN). |
| `systemd/*.service` | Servicios que arrancan los dos scripts al prender la Pi. |

## Tópicos MQTT

| Tópico | Contenido |
|---|---|
| `tapitas/clasificacion` | JSON `{color, hue, saturation, value, prob_rota, sesion_id, pieza_id}` |
| `tapitas/imagen/<sesion_id>` | JPEG de la tapita |
| `tapitas/color_ia/<sesion_id>` | JSON `{color, origen, pieza_id}` — color según Gemini, para cada tapita |
| `tapitas/veredicto_ia/<sesion_id>` | JSON `{veredicto, razon, pieza_id}` — segunda opinión de rotura, solo para las dudosas |

`pieza_id` asocia los resultados de la IA (que llegan segundos después) a
su tapita, aunque ya haya pasado otra. El panel muestra y cuenta el color
de la IA cuando está, y si no el del ESP32.

## Instalación

Todo sale de apt, sin venv:

```sh
sudo apt install mosquitto python3-paho-mqtt python3-fastapi python3-uvicorn python3-websockets
mkdir -p ~/tapitas
cp tapitas_ingest.py tapitas_dashboard.py dashboard.html ~/tapitas/
sudo cp systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tapitas-ingest tapitas-dashboard
```

Los `.service` asumen el usuario `sidler` y la carpeta `/home/sidler/tapitas`:
ajustarlos si cambia. Mosquitto tiene que aceptar conexiones de la red local
(no solo de `localhost`) para que la PC pueda publicar.

Panel: `http://<ip-de-la-raspberry>:8000`.

## Variables de entorno (opcionales)

| Variable | Default | Uso |
|---|---|---|
| `TAPITAS_BASE_DIR` | `~/tapitas` | Dónde viven la DB y las fotos |
| `TAPITAS_MQTT_HOST` / `TAPITAS_MQTT_PORT` | `127.0.0.1` / `1883` | Broker (ingest) |
| `TAPITAS_UMBRAL_ROTA` | `0.20` | Umbral sana/rota sobre `prob_rota` (ver [`modelo/threshold_results.txt`](../modelo/threshold_results.txt)) |
| `TAPITAS_CONFIANZA_MINIMA` | `0.50` | Debajo de esta confianza la tapita queda "dudoso" |
| `TAPITAS_DASH_PORT` | `8000` | Puerto del panel |
