#!/usr/bin/env python3
"""
Receptor para el protocolo de imx_serial_img (usado por el proyecto IMX708).
Formato:
    IMGSTART name=<n> fmt=<jpeg|rgb565> w=<w> h=<h> len=<N> crc32=<hex> [k=v ...]
    <exactamente N bytes crudos>
    IMGEND

Este script hace tres cosas a la vez:
  1. Recibe cada foto por serial, la muestra en UNA sola ventana que se va
     actualizando (OpenCV) y la guarda en disco -- ya sea en captura manual
     ('c'/'i' en la placa), en streaming ('v', ~1 foto cada 1-2s) o en el
     disparo automatico por sensor IR + cinta ('a': la placa detecta la
     tapita, frena, espera que se asiente y dispara sola).
  2. Reproduce por consola cualquier otra linea que mande el ESP32 -- boot
     log, resultado de color (HSV), estado sana/rota del modelo TinyML
     (prob_rota), etc. La lista completa de comandos del firmware la
     imprime la propia placa al arrancar (no se duplica aca a proposito,
     para no terminar con una copia vieja: reiniciala con el boton RST si
     te la perdiste).
  3. Actua de puente MQTT hacia la Raspberry: cada clasificacion
     (MQTTDATA ... color + prob_rota) y la imagen correspondiente se
     republican por MQTT, ver handle_mqttdata_line() mas abajo.

Comandos de ESTE script (no se mandan al ESP32): 'sana' / 'rota' activan
el modo dataset (cada foto que saques cae sola en esa carpeta, numerada),
'dataset off' lo apaga, 'salir' corta el script. Cualquier otra cosa que
escribas se manda tal cual por serial a la placa.
"""
import serial
import threading
import zlib
import os
from datetime import datetime

import numpy as np
import cv2
import paho.mqtt.client as mqtt
import json

PORT = "/dev/ttyACM0"
BAUD = 2000000
SAVE_DIR = os.path.expanduser("~/Documentos/capturas_imx708")

# Modo dataset: escribiendo "rota" o "sana" en la terminal, cada foto que
# saques con 'i' desde ahi en mas cae sola en la carpeta correspondiente,
# numerada. No hace falta renombrar nada a mano.
DATASET_DIR = os.path.expanduser("~/Documentos/dataset_tapitas")
dataset_label = None      # None, "sana" o "rota"
dataset_counters = {"sana": 0, "rota": 0}

for _label in ("sana", "rota"):
    _dir = os.path.join(DATASET_DIR, _label)
    os.makedirs(_dir, exist_ok=True)
    _existentes = [f for f in os.listdir(_dir) if f.endswith(".jpg")]
    dataset_counters[_label] = len(_existentes)
WINDOW_NAME = "IMX708 - vista en vivo"

# Puente hacia la Raspberry: el ESP32 ya no toca WiFi (bug de esp-hosted-mcu
# que reiniciaba la placa). Esta PC recibe la clasificacion por el cable
# serial de siempre y la republica por MQTT, sin pasar por el C6 para nada.
MQTT_BROKER_HOST = "192.168.1.100"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC_CLASIFICACION = "tapitas/clasificacion"

# Debe coincidir con SAMPLE_WINDOW del firmware (imx708_snapshot_main.c).
# Se usa solo para dibujar el rectangulo de referencia, no afecta la captura.
SAMPLE_WINDOW = 300

os.makedirs(SAVE_DIR, exist_ok=True)
ser = serial.Serial(PORT, BAUD, timeout=2)

# Cliente MQTT hacia la Raspberry. loop_start() corre en su propio hilo y
# reconecta solo si el broker se cae -- no bloquea la lectura del serial.
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="ver_imx708-puente")


def on_mqtt_connect(client, userdata, flags, reason_code, properties):
    if reason_code.is_failure:
        print(f"[MQTT] conexion rechazada: {reason_code}")
    else:
        print(f"[MQTT] conectado al broker {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")


def on_mqtt_disconnect(client, userdata, flags, reason_code, properties):
    print(f"[MQTT] desconectado ({reason_code}) -- reintentando solo")


mqtt_client.on_connect = on_mqtt_connect
mqtt_client.on_disconnect = on_mqtt_disconnect
mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)
try:
    mqtt_client.connect_async(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=30)
    mqtt_client.loop_start()
except Exception as e:
    print(f"[MQTT] no pude iniciar la conexion: {e}")

# La imagen (protocolo IMGSTART/IMGEND) llega SIEMPRE un instante antes que
# la linea MQTTDATA con el sesion_id (asi esta ordenado en el firmware: se
# manda la foto por serial, y recien despues se emite la clasificacion).
# Se guarda aca hasta que llegue esa linea y sepamos a que sesion pertenece.
pending_image_bytes = None


def handle_mqttdata_line(line):
    """Recibe 'MQTTDATA {json}' del ESP32 por serial y lo republica por MQTT."""
    global pending_image_bytes
    payload = line[len("MQTTDATA "):].strip()
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        print(f"[MQTT] JSON invalido del ESP32, descartado: {e} -- linea: {payload!r}")
        return
    mqtt_client.publish(MQTT_TOPIC_CLASIFICACION, json.dumps(data), qos=1)
    print(f"[MQTT] publicado: {data}")

    if pending_image_bytes is not None:
        sesion = data.get("sesion_id") or "_sin_sesion"
        topic = f"tapitas/imagen/{sesion}"
        mqtt_client.publish(topic, pending_image_bytes, qos=1)
        print(f"[MQTT] imagen ({len(pending_image_bytes)} bytes) publicada en {topic}")
        pending_image_bytes = None


def parse_header(line):
    """Convierte 'IMGSTART name=x fmt=jpeg w=100 h=100 len=500 crc32=abcd' en dict."""
    parts = line.strip().split()
    header = {}
    for token in parts[1:]:  # se salta el "IMGSTART"
        if "=" in token:
            k, v = token.split("=", 1)
            header[k] = v
    return header


def show_image(img_bgr):
    """Muestra la imagen en la ventana persistente (la crea si todavia no existe)."""
    cv2.imshow(WINDOW_NAME, img_bgr)
    # waitKey(1) es necesario para que la ventana efectivamente pinte y
    # procese eventos (mover/redimensionar). No bloquea el loop porque
    # el timeout es de solo 1ms.
    cv2.waitKey(1)


def draw_sample_window(img_bgr):
    """Dibuja el rectangulo de 300x300 centrado que el firmware usa para
    muestrear el color (SAMPLE_WINDOW). Sirve para chequear a ojo, de forma
    objetiva, si la tapita cae adentro de esa ventana o se sale."""
    h, w = img_bgr.shape[:2]
    x0 = (w - SAMPLE_WINDOW) // 2
    y0 = (h - SAMPLE_WINDOW) // 2
    out = img_bgr.copy()
    cv2.rectangle(out, (x0, y0), (x0 + SAMPLE_WINDOW, y0 + SAMPLE_WINDOW),
                  (0, 0, 255), 3)  # rojo en BGR, bien visible
    return out


def decode_rgb565(raw, w, h):
    """Convierte RGB565 crudo a un array BGR (formato que espera OpenCV)."""
    arr = np.frombuffer(raw, dtype=np.uint16).reshape(h, w)
    r = ((arr >> 11) & 0x1F) * 255 // 31
    g = ((arr >> 5) & 0x3F) * 255 // 63
    b = (arr & 0x1F) * 255 // 31
    # OpenCV espera BGR, no RGB
    bgr = np.stack([b, g, r], axis=-1).astype("uint8")
    return bgr


def handle_image(header):
    global pending_image_bytes
    name = header.get("name", "frame")
    fmt = header.get("fmt", "jpeg")
    w = int(header.get("w", 0))
    h = int(header.get("h", 0))
    length = int(header.get("len", 0))
    expected_crc = header.get("crc32", "").lower()

    print(f"  [recibiendo {fmt} {w}x{h}, {length} bytes...]")

    raw = b""
    while len(raw) < length:
        chunk = ser.read(length - len(raw))
        if not chunk:
            print("  [ERROR] se cortó la transferencia, faltaron bytes")
            return
        raw += chunk

    # Verificar el CRC32
    actual_crc = format(zlib.crc32(raw) & 0xFFFFFFFF, "08x")
    if expected_crc and actual_crc != expected_crc:
        print(f"  [ERROR] CRC no coincide! esperado={expected_crc} real={actual_crc} -> imagen corrupta, descartada")
        return
    else:
        print(f"  [OK] CRC verificado ({actual_crc})")

    # Confirmar el marcador de cierre
    closing = ser.readline().decode(errors="ignore").strip()
    if closing != "IMGEND":
        print(f"  [aviso] se esperaba IMGEND, se recibió: '{closing}'")

    ts = datetime.now().strftime("%H%M%S")

    if fmt == "jpeg":
        path = os.path.join(SAVE_DIR, f"{name}_{ts}.jpg")
        with open(path, "wb") as f:
            f.write(raw)
        last_path = os.path.join(SAVE_DIR, "ultima.jpg")
        with open(last_path, "wb") as f:
            f.write(raw)

        if dataset_label:
            dataset_counters[dataset_label] += 1
            n = dataset_counters[dataset_label]
            dataset_path = os.path.join(DATASET_DIR, dataset_label, f"{dataset_label}_{n:04d}.jpg")
            with open(dataset_path, "wb") as f:
                f.write(raw)
            print(f"  [dataset] guardada en {dataset_label}/ (#{n})")

        # Se guarda como pendiente: la linea MQTTDATA con el sesion_id llega
        # justo despues, y ahi se publica esta misma imagen hacia la Pi.
        pending_image_bytes = raw
        print(f"  [guardado] {path}")

        img_array = np.frombuffer(raw, dtype=np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            show_image(draw_sample_window(img_bgr))
        else:
            print("  [aviso] no se pudo decodificar el JPEG para mostrarlo")

    elif fmt == "rgb565":
        path = os.path.join(SAVE_DIR, f"{name}_{ts}.png")
        img_bgr = decode_rgb565(raw, w, h)
        cv2.imwrite(path, img_bgr)
        print(f"  [guardado] {path}")
        show_image(draw_sample_window(img_bgr))

    else:
        print(f"  [aviso] formato desconocido '{fmt}', guardando crudo")
        with open(os.path.join(SAVE_DIR, f"{name}_{ts}.raw"), "wb") as f:
            f.write(raw)


def serial_reader():
    while True:
        try:
            line = ser.readline().decode(errors="ignore").rstrip("\r\n")
        except Exception as e:
            print(f"[error de lectura serial: {e}]")
            continue

        if not line:
            continue

        if line.startswith("IMGSTART"):
            header = parse_header(line)
            handle_image(header)
            continue

        if line.startswith("MQTTDATA "):
            handle_mqttdata_line(line)
            continue

        # Cualquier otra línea es log normal del ESP32 (incluye el resultado de color)
        print(line)


threading.Thread(target=serial_reader, daemon=True).start()

print(f"Conectado a {PORT} a {BAUD} baudios.")
print("La lista de comandos del firmware la imprime la propia placa al bootear")
print("(reiniciala con RST si te la perdiste). Comandos de este script:")
print("  sana / rota   -> modo dataset: cada foto capturada cae sola en esa carpeta")
print("  dataset off   -> apaga el modo dataset")
print("  salir         -> corta el script (cualquier otra cosa se manda tal cual al ESP32)")
print("La ventana de imagen se actualiza sola con streaming ('v') o disparo automatico ('a').")
print(f"Clasificacion (color + prob_rota) puenteada por MQTT a {MQTT_BROKER_HOST}, "
      f"topico '{MQTT_TOPIC_CLASIFICACION}'.\n")

while True:
    cmd = input().strip()
    low = cmd.lower()

    if low == "salir":
        break

    if low in ("sana", "rota"):
        dataset_label = low
        print(f"[dataset] modo activado: {dataset_label} "
              f"(ya hay {dataset_counters[dataset_label]} fotos guardadas)")
        continue

    if low == "dataset off":
        dataset_label = None
        print("[dataset] modo apagado")
        continue

    ser.write((cmd + "\n").encode())

cv2.destroyAllWindows()
mqtt_client.loop_stop()
