#!/usr/bin/env python3
"""
tapitas_ingest.py - Suscriptor MQTT que persiste clasificaciones de tapitas en
SQLite y guarda las imagenes JPEG asociadas.

Topicos:
  tapitas/clasificacion        JSON: {color, hue, saturation, value, prob_rota, sesion_id}
  tapitas/imagen               JPEG binario -> se asocia al ultimo registro
  tapitas/imagen/<sesion_id>   JPEG binario -> se asocia al ultimo registro de esa sesion
  tapitas/veredicto_ia/<sesion_id>  JSON: {veredicto, razon, pieza_id} -> segunda opinion
                                     de un modelo de vision (ver ver_imx708.py) para casos
                                     "dudoso"; se guarda aparte, no pisa el estado
  tapitas/color_ia/<sesion_id>      JSON: {color, origen, pieza_id} -> color segun un
                                     modelo de vision, para cada tapita; se guarda aparte
                                     del color del ESP32 (que queda como respaldo)

pieza_id lo agrega ver_imx708.py a cada clasificacion: los mensajes de IA llegan
segundos despues, y con pieza_id se asocian a su tapita aunque ya haya pasado
otra. Sin pieza_id (mensajes viejos) se usa el ultimo registro de la sesion.

Broker local, sin autenticacion. paho-mqtt 2.x + sqlite3 (ambos de la stdlib/apt,
sin venv ni Docker).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import paho.mqtt.client as mqtt

# --------------------------------------------------------------------------- #
# Configuracion (ajustable por variables de entorno)
# --------------------------------------------------------------------------- #
MQTT_HOST = os.environ.get("TAPITAS_MQTT_HOST", "127.0.0.1")
MQTT_PORT = int(os.environ.get("TAPITAS_MQTT_PORT", "1883"))
MQTT_KEEPALIVE = 60

# Umbral sana/rota sobre prob_rota (0.0-1.0), ver modelo/threshold_results.txt:
# a 0.20 el modelo da 90.9% accuracy / 95.7% recall de rotas en el set de
# validacion. El ESP32 manda prob_rota cruda a proposito -- el umbral se
# aplica aca para poder ajustarlo sin reflashear la placa.
UMBRAL_ROTA = float(os.environ.get("TAPITAS_UMBRAL_ROTA", "0.20"))

# Los falsos positivos del set de validacion (sana clasificada como rota) caen
# justo en prob_rota ~0.15-0.25, pegados al umbral -- son el caso de una
# tapita sana con logo/texto en relieve (ver tapitas de Coca-Cola), no un
# error de pipeline. En vez de forzar una decision poco confiable, cualquier
# clasificacion cuya confianza (la misma que se muestra en el dashboard)
# quede por debajo de este piso se marca "dudoso" para que la revise Ollama.
CONFIANZA_MINIMA_DUDOSO = float(os.environ.get("TAPITAS_CONFIANZA_MINIMA", "0.50"))


def es_dudoso(prob_rota: float) -> bool:
    confianza = prob_rota if prob_rota > UMBRAL_ROTA else (1 - prob_rota)
    return confianza < CONFIANZA_MINIMA_DUDOSO

BASE_DIR = Path(os.environ.get("TAPITAS_BASE_DIR", str(Path.home() / "tapitas")))
DB_PATH = BASE_DIR / "clasificacion.db"
IMG_DIR = BASE_DIR / "imagenes"

TOPIC_CLASIFICACION = "tapitas/clasificacion"
TOPIC_IMAGEN = "tapitas/imagen"
TOPIC_IMAGEN_SESION = "tapitas/imagen/+"
TOPIC_VEREDICTO_IA = "tapitas/veredicto_ia/+"
TOPIC_COLOR_IA = "tapitas/color_ia/+"
COLORES_IA = ("naranja", "rojo", "amarillo", "azul", "lila", "blanco", "rosa")

MAX_IMG_BYTES = 8 * 1024 * 1024  # descarta payloads absurdos

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("tapitas")

# --------------------------------------------------------------------------- #
# Base de datos
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS clasificacion (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    sesion_id   TEXT,
    color       TEXT,
    hue         REAL,
    saturation  REAL,
    value       REAL,
    prob_rota   REAL,
    estado      TEXT,
    imagen_path TEXT,
    veredicto_ia       TEXT,
    veredicto_ia_razon TEXT,
    pieza_id    TEXT,
    color_ia    TEXT,
    color_ia_origen TEXT
);
CREATE INDEX IF NOT EXISTS idx_clasificacion_sesion ON clasificacion (sesion_id);
"""

# paho llama a los callbacks desde un solo hilo (loop_forever), pero el lock deja
# el acceso a SQLite seguro si en el futuro se pasa a loop_start().
_db_lock = threading.Lock()


def db_connect() -> sqlite3.Connection:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(SCHEMA)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(clasificacion)")}
    if "prob_rota" not in cols:
        conn.execute("ALTER TABLE clasificacion ADD COLUMN prob_rota REAL")
    if "veredicto_ia" not in cols:
        conn.execute("ALTER TABLE clasificacion ADD COLUMN veredicto_ia TEXT")
    if "veredicto_ia_razon" not in cols:
        conn.execute("ALTER TABLE clasificacion ADD COLUMN veredicto_ia_razon TEXT")
    for col in ("pieza_id", "color_ia", "color_ia_origen"):
        if col not in cols:
            conn.execute(f"ALTER TABLE clasificacion ADD COLUMN {col} TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_clasificacion_pieza ON clasificacion (pieza_id)")
    conn.commit()
    return conn


def _fs_safe(ts_iso: str) -> str:
    # 2026-09-10T21:45:03.123456+00:00 -> 2026-09-10T21-45-03.123456Z
    return ts_iso.replace(":", "-").replace("+00-00", "Z")


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #
def handle_clasificacion(conn: sqlite3.Connection, payload: bytes) -> None:
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        log.warning("clasificacion: JSON invalido (%s) - descartado", exc)
        return
    if not isinstance(data, dict):
        log.warning("clasificacion: payload no es objeto JSON - descartado")
        return

    prob_rota = data.get("prob_rota")
    estado = None
    if isinstance(prob_rota, (int, float)):
        if es_dudoso(prob_rota):
            estado = "dudoso"
        else:
            estado = "rota" if prob_rota > UMBRAL_ROTA else "sana"

    ts = datetime.now(timezone.utc).isoformat()
    row = (
        ts,
        data.get("sesion_id"),
        data.get("color"),
        data.get("hue"),
        data.get("saturation"),
        data.get("value"),
        prob_rota,
        estado,
        data.get("pieza_id"),
    )
    with _db_lock:
        cur = conn.execute(
            "INSERT INTO clasificacion "
            "(timestamp, sesion_id, color, hue, saturation, value, prob_rota, estado, pieza_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            row,
        )
        conn.commit()
        rid = cur.lastrowid
    log.info(
        "clasificacion #%s sesion=%s color=%s hsv=(%s,%s,%s) prob_rota=%s estado=%s",
        rid, data.get("sesion_id"), data.get("color"),
        data.get("hue"), data.get("saturation"), data.get("value"),
        prob_rota, estado,
    )


def _target_row(conn: sqlite3.Connection, sesion_id: str | None):
    """Registro al que se asocia una imagen entrante."""
    with _db_lock:
        if sesion_id:
            r = conn.execute(
                "SELECT id, sesion_id, timestamp FROM clasificacion "
                "WHERE sesion_id = ? ORDER BY id DESC LIMIT 1",
                (sesion_id,),
            ).fetchone()
            if r:
                return r
        return conn.execute(
            "SELECT id, sesion_id, timestamp FROM clasificacion "
            "ORDER BY id DESC LIMIT 1"
        ).fetchone()


def _target_row_ia(conn: sqlite3.Connection, pieza_id, sesion_id: str):
    """Registro al que se asocia un mensaje de IA: por pieza_id si viene,
    si no (mensajes viejos) el ultimo de la sesion."""
    if pieza_id:
        with _db_lock:
            r = conn.execute(
                "SELECT id, sesion_id, timestamp FROM clasificacion WHERE pieza_id = ?",
                (pieza_id,),
            ).fetchone()
        if r:
            return r
    return _target_row(conn, sesion_id)


def handle_imagen(conn: sqlite3.Connection, payload: bytes,
                  sesion_id_topico: str | None) -> None:
    if not payload:
        log.warning("imagen: payload vacio - descartado")
        return
    if len(payload) > MAX_IMG_BYTES:
        log.warning("imagen: %d bytes supera el maximo - descartado", len(payload))
        return
    if payload[:2] != b"\xff\xd8":
        log.warning("imagen: no empieza con magic bytes JPEG - se guarda igual")

    row = _target_row(conn, sesion_id_topico)
    now_iso = datetime.now(timezone.utc).isoformat()

    if row is None:
        sesion = sesion_id_topico or "_sin_registro"
        dest = IMG_DIR / sesion / f"{_fs_safe(now_iso)}.jpg"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(payload)
        log.warning("imagen (%d bytes) sin registro asociado -> %s",
                    len(payload), dest)
        return

    rid, sesion_row, ts_row = row
    sesion = sesion_row or sesion_id_topico or "_sin_sesion"
    dest = IMG_DIR / str(sesion) / f"{_fs_safe(ts_row or now_iso)}.jpg"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)

    with _db_lock:
        conn.execute(
            "UPDATE clasificacion SET imagen_path = ? WHERE id = ?",
            (str(dest), rid),
        )
        conn.commit()
    log.info("imagen (%d bytes) -> %s  (registro #%s)", len(payload), dest, rid)


def handle_veredicto_ia(conn: sqlite3.Connection, payload: bytes,
                         sesion_id: str) -> None:
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        log.warning("veredicto_ia: JSON invalido (%s) - descartado", exc)
        return
    veredicto = data.get("veredicto")
    if veredicto not in ("sana", "rota"):
        log.warning("veredicto_ia: valor invalido %r - descartado", veredicto)
        return
    razon = data.get("razon") or ""

    row = _target_row_ia(conn, data.get("pieza_id"), sesion_id)
    if row is None:
        log.warning("veredicto_ia para sesion=%s sin registro asociado", sesion_id)
        return
    rid = row[0]

    with _db_lock:
        conn.execute(
            "UPDATE clasificacion SET veredicto_ia = ?, veredicto_ia_razon = ? "
            "WHERE id = ?",
            (veredicto, razon, rid),
        )
        conn.commit()
    log.info("veredicto_ia sesion=%s -> %s (%s)  (registro #%s)",
              sesion_id, veredicto, razon[:80], rid)


def handle_color_ia(conn: sqlite3.Connection, payload: bytes,
                    sesion_id: str) -> None:
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        log.warning("color_ia: JSON invalido (%s) - descartado", exc)
        return
    color = data.get("color")
    if color not in COLORES_IA:
        log.warning("color_ia: valor invalido %r - descartado", color)
        return
    origen = data.get("origen") or ""

    row = _target_row_ia(conn, data.get("pieza_id"), sesion_id)
    if row is None:
        log.warning("color_ia para sesion=%s sin registro asociado", sesion_id)
        return
    rid = row[0]

    with _db_lock:
        conn.execute(
            "UPDATE clasificacion SET color_ia = ?, color_ia_origen = ? WHERE id = ?",
            (color, origen, rid),
        )
        conn.commit()
    log.info("color_ia sesion=%s -> %s (%s)  (registro #%s)", sesion_id, color, origen, rid)


# --------------------------------------------------------------------------- #
# Callbacks MQTT (paho CallbackAPIVersion.VERSION2)
# --------------------------------------------------------------------------- #
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code.is_failure:
        log.error("conexion MQTT rechazada: %s", reason_code)
        return
    log.info("conectado al broker %s:%s", MQTT_HOST, MQTT_PORT)
    client.subscribe([
        (TOPIC_CLASIFICACION, 1),
        (TOPIC_IMAGEN, 1),
        (TOPIC_IMAGEN_SESION, 1),
        (TOPIC_VEREDICTO_IA, 1),
        (TOPIC_COLOR_IA, 1),
    ])
    log.info("suscripto a %s | %s | %s | %s | %s",
             TOPIC_CLASIFICACION, TOPIC_IMAGEN, TOPIC_IMAGEN_SESION,
             TOPIC_VEREDICTO_IA, TOPIC_COLOR_IA)


def on_disconnect(client, userdata, flags, reason_code, properties):
    if reason_code.is_failure:
        log.warning("desconexion inesperada (%s) - paho reintentara", reason_code)
    else:
        log.info("desconectado limpiamente")


def on_message(client, userdata, msg):
    conn = userdata
    try:
        if msg.topic == TOPIC_CLASIFICACION:
            handle_clasificacion(conn, msg.payload)
        elif msg.topic == TOPIC_IMAGEN:
            handle_imagen(conn, msg.payload, None)
        elif msg.topic.startswith(TOPIC_IMAGEN + "/"):
            sesion = msg.topic.split("/", 2)[2]
            handle_imagen(conn, msg.payload, sesion)
        elif msg.topic.startswith("tapitas/veredicto_ia/"):
            sesion = msg.topic.split("/", 2)[2]
            handle_veredicto_ia(conn, msg.payload, sesion)
        elif msg.topic.startswith("tapitas/color_ia/"):
            sesion = msg.topic.split("/", 2)[2]
            handle_color_ia(conn, msg.payload, sesion)
        else:
            log.debug("topico ignorado: %s", msg.topic)
    except Exception:
        log.exception("error procesando mensaje en %s", msg.topic)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    conn = db_connect()
    log.info("SQLite: %s", DB_PATH)

    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="tapitas-ingest",
        userdata=conn,
    )
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    def _term(signum, _frame):
        log.info("senal %s recibida - cerrando", signum)
        client.disconnect()

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)

    client.connect_async(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE)
    try:
        client.loop_forever(retry_first_connection=True)
    finally:
        conn.close()
        log.info("terminado")
    return 0


if __name__ == "__main__":
    sys.exit(main())
