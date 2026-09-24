#!/usr/bin/env python3
"""
app_tapitas.py - Version de escritorio (PyQt6) del dashboard web de tapitas.

Escucha el broker MQTT (clasificacion, imagen y veredicto de la IA) y guarda
todo en su propia base SQLite local, asi el historial no depende de la
Raspberry. Usa el mismo .env que ver_imx708.py: con TAPITAS_MQTT_HOST=localhost
y Mosquitto corriendo en esta PC funciona sin red ni Raspberry.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime

import paho.mqtt.client as mqtt
from dotenv import load_dotenv
from PyQt6.QtCore import QObject, Qt, pyqtSignal as Signal
from PyQt6.QtGui import QColor, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel,
    QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))
try:
    MQTT_HOST = os.environ["TAPITAS_MQTT_HOST"]
except KeyError:
    raise SystemExit("Falta TAPITAS_MQTT_HOST en el .env (ver .env.example).")
DB_PATH = os.environ.get("TAPITAS_LOCAL_DB", os.path.join(BASE_DIR, "tapitas_local.db"))

# Mismos criterios que ver_imx708.py / tapitas_ingest.py.
UMBRAL_ROTA = float(os.environ.get("TAPITAS_UMBRAL_ROTA", "0.20"))
CONFIANZA_MINIMA_DUDOSO = float(os.environ.get("TAPITAS_CONFIANZA_MINIMA", "0.50"))

COLORES = ["naranja", "rojo", "amarillo", "azul", "lila", "blanco"]
HEX = {
    "naranja": "#ff7a18", "rojo": "#e5352b", "amarillo": "#f4c400",
    "azul": "#2d7ff9", "lila": "#b57edc", "blanco": "#f5f5f5", "otro": "#6b7a8d",
}
COLOR_ESTADO = {"sana": "#35c988", "rota": "#e5533d", "dudoso": "#f4c400"}
MAX_FILAS = 200

STYLE = """
QWidget { background: #0b0f14; color: #d7e0ea; font-family: monospace; font-size: 13px; }
QFrame#panel { background: #131a22; border: 1px solid #24303c; border-radius: 6px; }
QLabel { background: transparent; }
QLabel#contador { font-size: 34px; font-weight: bold; }
QLabel#titulo { font-size: 16px; font-weight: bold; letter-spacing: 3px; }
QLabel#k { color: #7d8ea0; font-size: 11px; }
QLabel#seccion { color: #7d8ea0; font-size: 11px; letter-spacing: 2px; }
QTableWidget { background: #131a22; gridline-color: #1a232d; border: none; }
QHeaderView::section { background: #131a22; color: #7d8ea0; border: none;
                       border-bottom: 1px solid #24303c; padding: 6px; font-size: 11px; }
"""


def estado_de(prob_rota):
    if not isinstance(prob_rota, (int, float)):
        return None
    confianza = prob_rota if prob_rota > UMBRAL_ROTA else (1 - prob_rota)
    if confianza < CONFIANZA_MINIMA_DUDOSO:
        return "dudoso"
    return "rota" if prob_rota > UMBRAL_ROTA else "sana"


def confianza_txt(row):
    p = row.get("prob_rota")
    if not isinstance(p, (int, float)):
        return "—"
    if row.get("estado") == "dudoso":
        return f"{round(p * 100)}% rota"
    c = p if row.get("estado") == "rota" else 1 - p
    return f"{round(c * 100)}%"


def num(v):
    return "—" if v is None else f"{v:.1f}"


def norm_color(c):
    c = (c or "").strip().lower()
    return c if c in COLORES else "otro"


SCHEMA = """
CREATE TABLE IF NOT EXISTS clasificacion (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp          TEXT NOT NULL,
    sesion_id          TEXT,
    color              TEXT,
    hue                REAL,
    saturation         REAL,
    value              REAL,
    prob_rota          REAL,
    estado             TEXT,
    imagen             BLOB,
    veredicto_ia       TEXT,
    veredicto_ia_razon TEXT
);
"""
COLS_SIN_IMAGEN = ("id, timestamp, sesion_id, color, hue, saturation, value, "
                   "prob_rota, estado, veredicto_ia, veredicto_ia_razon")


def db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


class Bus(QObject):
    """Pasa eventos del hilo de MQTT al hilo de la UI (donde vive la DB)."""
    nueva = Signal(dict)
    imagen = Signal(str, bytes)
    veredicto = Signal(str, dict)
    conexion = Signal(bool)


class Barras(QWidget):
    def __init__(self):
        super().__init__()
        self.valores = {c: 0 for c in COLORES}
        self.setMinimumHeight(220)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        base = h - 24
        maximo = max(max(self.valores.values()), 1)
        ancho = w / len(COLORES)
        for i, c in enumerate(COLORES):
            v = self.valores[c]
            alto = (base - 20) * v / maximo
            x = i * ancho + ancho * 0.2
            p.fillRect(int(x), int(base - alto), int(ancho * 0.6), int(alto), QColor(HEX[c]))
            p.setPen(QColor("#9fb0c0"))
            p.drawText(int(i * ancho), base + 4, int(ancho), 20, Qt.AlignmentFlag.AlignCenter, c.capitalize())
            p.drawText(int(i * ancho), int(base - alto - 18), int(ancho), 16, Qt.AlignmentFlag.AlignCenter, str(v))


def panel(titulo):
    f = QFrame(objectName="panel")
    lay = QVBoxLayout(f)
    lay.addWidget(QLabel(titulo.upper(), objectName="seccion"))
    return f, lay


class Ventana(QMainWindow):
    COLS = ["ID", "Hora", "Color", "H", "S", "V", "Estado", "Confianza", "Veredicto IA", "Sesión"]

    def __init__(self, bus: Bus, db: sqlite3.Connection):
        super().__init__()
        self.setWindowTitle("Control de Calidad · Tapitas")
        self.resize(1250, 850)
        self.db = db
        self.filas: list[dict] = []
        self.conteos = {c: 0 for c in COLORES}
        self.total = 0

        central = QWidget()
        root = QVBoxLayout(central)
        self.setCentralWidget(central)

        header = QHBoxLayout()
        header.addWidget(QLabel("CONTROL DE CALIDAD", objectName="titulo"))
        header.addStretch()
        self.lbl_total = QLabel("Piezas inspeccionadas: 0")
        self.lbl_conexion = QLabel("● conectando")
        self.lbl_conexion.setStyleSheet("color:#e5533d")
        header.addWidget(self.lbl_total)
        header.addSpacing(20)
        header.addWidget(self.lbl_conexion)
        root.addLayout(header)

        # contadores
        f, lay = panel("Conteo por color")
        fila = QHBoxLayout()
        self.lbl_conteo = {}
        for c in COLORES:
            card = QFrame(objectName="panel")
            card.setStyleSheet(f"QFrame#panel {{ border-top: 3px solid {HEX[c]}; }}")
            cl = QVBoxLayout(card)
            nombre = QLabel(c.upper(), objectName="k")
            n = QLabel("0", objectName="contador")
            cl.addWidget(nombre)
            cl.addWidget(n)
            self.lbl_conteo[c] = n
            fila.addWidget(card)
        lay.addLayout(fila)
        root.addWidget(f)

        # grafico + ultima pieza
        medio = QHBoxLayout()
        f, lay = panel("Distribución")
        self.barras = Barras()
        lay.addWidget(self.barras)
        medio.addWidget(f, 1)

        f, lay = panel("Última pieza inspeccionada")
        ult = QHBoxLayout()
        self.foto = QLabel("sin imagen")
        self.foto.setFixedSize(240, 240)
        self.foto.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.foto.setStyleSheet("background:#0e141b; border:1px solid #24303c; color:#7d8ea0")
        ult.addWidget(self.foto)
        datos = QGridLayout()
        self.lbl_color = QLabel("—")
        datos.addWidget(self.lbl_color, 0, 0, 1, 2)
        self.campos = {}
        for i, (clave, nombre) in enumerate([
            ("id", "ID"), ("hora", "Hora"), ("sesion", "Sesión"), ("estado", "Estado"),
            ("confianza", "Confianza"), ("veredicto", "Veredicto IA"),
            ("hsv", "H / S / V"), ("razon", "Razón IA"),
        ], start=1):
            datos.addWidget(QLabel(nombre.upper(), objectName="k"), i, 0)
            v = QLabel("—")
            v.setWordWrap(True)
            datos.addWidget(v, i, 1)
            self.campos[clave] = v
        datos.setRowStretch(len(self.campos) + 1, 1)
        ult.addLayout(datos, 1)
        lay.addLayout(ult)
        medio.addWidget(f, 1)
        root.addLayout(medio)

        # historial
        f, lay = panel("Historial de inspecciones")
        self.tabla = QTableWidget(0, len(self.COLS))
        self.tabla.setHorizontalHeaderLabels(self.COLS)
        self.tabla.verticalHeader().setVisible(False)
        self.tabla.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tabla.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tabla.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        lay.addWidget(self.tabla)
        root.addWidget(f, 1)

        bus.nueva.connect(self.on_nueva)
        bus.imagen.connect(self.on_imagen)
        bus.veredicto.connect(self.on_veredicto)
        bus.conexion.connect(self.on_conexion)
        self.cargar_historial()

    # ---- render ----
    def render_conteos(self):
        for c in COLORES:
            self.lbl_conteo[c].setText(str(self.conteos[c]))
        self.barras.valores = dict(self.conteos)
        self.barras.update()
        self.lbl_total.setText(f"Piezas inspeccionadas: {self.total}")

    def celdas(self, r):
        hora = r.get("timestamp") or ""
        try:
            hora = datetime.fromisoformat(hora).astimezone().strftime("%d/%m %H:%M:%S")
        except ValueError:
            pass
        return [
            f"#{r['id']}", hora, r.get("color") or "—", num(r.get("hue")),
            num(r.get("saturation")), num(r.get("value")),
            (r.get("estado") or "—").upper(), confianza_txt(r),
            (r.get("veredicto_ia") or "—").upper(), r.get("sesion_id") or "—",
        ]

    def pintar_fila(self, idx, r):
        for col, txt in enumerate(self.celdas(r)):
            item = QTableWidgetItem(txt)
            if col in (3, 4, 5, 7):
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if col == 2:
                item.setForeground(QColor(HEX[norm_color(r.get("color"))]))
            if col == 6 and r.get("estado") in COLOR_ESTADO:
                item.setForeground(QColor(COLOR_ESTADO[r["estado"]]))
            if col == 8 and r.get("veredicto_ia") in COLOR_ESTADO:
                item.setForeground(QColor(COLOR_ESTADO[r["veredicto_ia"]]))
            self.tabla.setItem(idx, col, item)

    def render_ultima(self):
        if not self.filas:
            return
        r = self.filas[0]
        self.lbl_color.setText((r.get("color") or "—").capitalize())
        self.lbl_color.setStyleSheet(
            f"color:{HEX[norm_color(r.get('color'))]}; font-size:22px; font-weight:bold")
        self.campos["id"].setText(f"#{r['id']}")
        self.campos["hora"].setText(self.celdas(r)[1])
        self.campos["sesion"].setText(r.get("sesion_id") or "—")
        for clave, campo in (("estado", "estado"), ("veredicto", "veredicto_ia")):
            val = r.get(campo)
            self.campos[clave].setText((val or "—").upper())
            self.campos[clave].setStyleSheet(
                f"color:{COLOR_ESTADO.get(val, '#d7e0ea')}; font-weight:bold")
        self.campos["confianza"].setText(confianza_txt(r))
        self.campos["hsv"].setText(
            f"{num(r.get('hue'))} / {num(r.get('saturation'))} / {num(r.get('value'))}")
        self.campos["razon"].setText(r.get("veredicto_ia_razon") or "—")
        if r.get("_jpeg"):
            pix = QPixmap()
            pix.loadFromData(r["_jpeg"])
            self.foto.setPixmap(pix.scaled(240, 240, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                           Qt.TransformationMode.SmoothTransformation))
        else:
            self.foto.setText("sin imagen")

    def fila_de_sesion(self, sesion):
        for i, r in enumerate(self.filas):
            if r.get("sesion_id") == sesion:
                return i, r
        return None, None

    def cargar_historial(self):
        for fila in self.db.execute(
                "SELECT color, COUNT(*) AS n FROM clasificacion GROUP BY color"):
            c = norm_color(fila["color"])
            if c in self.conteos:
                self.conteos[c] += fila["n"]
            self.total += fila["n"]
        self.filas = [dict(r) for r in self.db.execute(
            f"SELECT {COLS_SIN_IMAGEN} FROM clasificacion ORDER BY id DESC LIMIT ?",
            (MAX_FILAS,))]
        if self.filas:
            img = self.db.execute("SELECT imagen FROM clasificacion WHERE id = ?",
                                  (self.filas[0]["id"],)).fetchone()
            self.filas[0]["_jpeg"] = img["imagen"]
        self.tabla.setRowCount(len(self.filas))
        for i, r in enumerate(self.filas):
            self.pintar_fila(i, r)
        self.render_conteos()
        self.render_ultima()

    # ---- eventos ----
    def on_nueva(self, data):
        r = {k: data.get(k) for k in
             ("sesion_id", "color", "hue", "saturation", "value", "prob_rota")}
        r["timestamp"] = datetime.now().astimezone().isoformat()
        r["estado"] = estado_de(r.get("prob_rota"))
        cur = self.db.execute(
            "INSERT INTO clasificacion (timestamp, sesion_id, color, hue, saturation, "
            "value, prob_rota, estado) VALUES (:timestamp, :sesion_id, :color, :hue, "
            ":saturation, :value, :prob_rota, :estado)", r)
        self.db.commit()
        r["id"] = cur.lastrowid
        self.filas.insert(0, r)
        del self.filas[MAX_FILAS:]
        self.tabla.insertRow(0)
        self.pintar_fila(0, r)
        if self.tabla.rowCount() > MAX_FILAS:
            self.tabla.removeRow(self.tabla.rowCount() - 1)
        c = norm_color(r.get("color"))
        if c in self.conteos:
            self.conteos[c] += 1
        self.total += 1
        self.render_conteos()
        self.render_ultima()

    def on_imagen(self, sesion, jpeg):
        i, r = self.fila_de_sesion(sesion)
        if r is None:
            return
        self.db.execute("UPDATE clasificacion SET imagen = ? WHERE id = ?", (jpeg, r["id"]))
        self.db.commit()
        if i == 0:
            r["_jpeg"] = jpeg
            self.render_ultima()

    def on_veredicto(self, sesion, data):
        # La ultima dudosa sin veredicto de la sesion, no la ultima fila: si
        # Ollama tarda y ya paso otra tapita, el veredicto no le corresponde.
        i, r = next(((i, r) for i, r in enumerate(self.filas)
                     if r.get("sesion_id") == sesion and r.get("estado") == "dudoso"
                     and not r.get("veredicto_ia")), (None, None))
        if r is None:
            return
        r["veredicto_ia"] = data.get("veredicto")
        r["veredicto_ia_razon"] = data.get("razon")
        self.db.execute(
            "UPDATE clasificacion SET veredicto_ia = ?, veredicto_ia_razon = ? WHERE id = ?",
            (r["veredicto_ia"], r["veredicto_ia_razon"], r["id"]))
        self.db.commit()
        self.pintar_fila(i, r)
        if i == 0:
            self.render_ultima()

    def on_conexion(self, ok):
        self.lbl_conexion.setText("● en vivo" if ok else "● desconectado")
        self.lbl_conexion.setStyleSheet(f"color:{'#35c988' if ok else '#e5533d'}")


def iniciar_mqtt(bus: Bus):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="tapitas-app-escritorio")

    def on_connect(c, _u, _f, rc, _p):
        bus.conexion.emit(not rc.is_failure)
        c.subscribe([("tapitas/clasificacion", 1), ("tapitas/imagen/+", 1),
                     ("tapitas/veredicto_ia/+", 1)])

    def on_disconnect(_c, _u, _f, _rc, _p):
        bus.conexion.emit(False)

    def on_message(_c, _u, msg):
        try:
            if msg.topic == "tapitas/clasificacion":
                bus.nueva.emit(json.loads(msg.payload))
            elif msg.topic.startswith("tapitas/imagen/"):
                bus.imagen.emit(msg.topic.split("/", 2)[2], bytes(msg.payload))
            elif msg.topic.startswith("tapitas/veredicto_ia/"):
                bus.veredicto.emit(msg.topic.split("/", 2)[2], json.loads(msg.payload))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[app] mensaje invalido en {msg.topic}: {e}")

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=15)
    client.connect_async(MQTT_HOST, 1883, 60)
    client.loop_start()
    return client


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    bus = Bus()
    db = db_connect()
    ventana = Ventana(bus, db)
    ventana.show()
    client = iniciar_mqtt(bus)
    code = app.exec()
    client.loop_stop()
    client.disconnect()
    db.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
