#!/usr/bin/env python3
"""
scada_tapitas.py - Pantalla de operador (SCADA) de la maqueta de tapitas.

Muestra la planta en vivo (cinta, sensor IR, camara, motor y las tapitas
viajando), manda los comandos al ESP32 y registra eventos y alarmas. Todo va
por MQTT a traves de ver_imx708.py, que queda como puente serial y consola de
diagnostico:

    ESP32 --USB-- ver_imx708.py --MQTT-- scada_tapitas.py

La pestana "Produccion" es la misma vista de app_tapitas.py (conteos,
historial) y usa la misma base local. No abrir las dos apps a la vez: cada una
guardaria cada tapita por su lado y quedarian duplicadas.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime

import paho.mqtt.client as mqtt
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal as Signal
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QPlainTextEdit, QPushButton,
    QSizePolicy, QTabWidget, QVBoxLayout, QWidget,
)

from app_tapitas import (
    COLOR_ESTADO, HEX, MQTT_HOST, STYLE, Bus, Ventana, color_final, db_connect,
    estado_de, norm_color, panel,
)

TOPIC_PUENTE = "tapitas/puente/estado"
TOPIC_PLANTA_ESTADO = "tapitas/planta/estado"
TOPIC_PLANTA_EVENTO = "tapitas/planta/evento"
TOPIC_PLANTA_LOG = "tapitas/planta/log"
TOPIC_PLANTA_COMANDO = "tapitas/planta/comando"

# Posiciones sobre la cinta (0 = rodillo de entrada, 1 = rodillo de salida),
# a ojo de las fotos de la maqueta: el tunel tapa casi toda la cinta, el
# sensor IR esta adentro, apenas pasando la boca, y la camara en el centro.
# Lo que tarda la cinta real en llevar algo de punta a punta, en segundos:
# marca la velocidad de la cinta y de las tapitas que salen. Del sensor a la
# camara la tapita tarda siempre el retardo de disparo, para que llegue bajo
# la camara justo cuando el ESP32 saca la foto.
TIEMPO_CRUCE_S = float(os.environ.get("TAPITAS_SCADA_CRUCE_S", "40"))
TUNEL = (0.24, 0.80)
POS_SENSOR = 0.28
POS_CAMARA = 0.52
# Cuanto espera una tapita bajo la camara a que llegue la foto antes de
# seguir viaje sin resultado (captura fallida, o disparo automatico apagado).
ESPERA_CAPTURA_S = 6.0

VERDE, ROJO, AMARILLO, CIAN = "#35c988", "#e5533d", "#f4c400", "#4fc3f7"
GRIS, GRIS_OSCURO, LINEA = "#7d8ea0", "#1a232d", "#3a4856"

STYLE_SCADA = STYLE + """
QTabWidget::pane { border: none; }
QTabBar::tab { background: #131a22; color: #7d8ea0; padding: 8px 18px;
               border: 1px solid #24303c; border-bottom: none; margin-right: 2px; }
QTabBar::tab:selected { color: #d7e0ea; background: #1a232d; }
QPushButton { background: #1a232d; border: 1px solid #2f3d4b; border-radius: 4px;
              padding: 7px 10px; color: #d7e0ea; }
QPushButton:hover { border-color: #4fc3f7; }
QPushButton:pressed { background: #24303c; }
QPushButton:disabled { color: #3f4b57; border-color: #1f2832; }
QPushButton#marcha { background: #123526; border-color: #35c988; color: #35c988;
                     font-size: 16px; font-weight: bold; padding: 14px; }
QPushButton#parada { background: #3a1714; border-color: #e5533d; color: #e5533d;
                     font-size: 16px; font-weight: bold; padding: 14px; }
QPushButton#marcha:disabled, QPushButton#parada:disabled {
                     background: #131a22; color: #3f4b57; border-color: #1f2832; }
QListWidget, QPlainTextEdit, QLineEdit { background: #0e141b; border: 1px solid #24303c; }
QLabel#valor { font-size: 15px; font-weight: bold; }
QLabel#estado_linea { font-size: 20px; font-weight: bold; letter-spacing: 2px; }
"""


def panel_fijo(titulo):
    """panel() de app_tapitas, con el titulo de alto fijo (que no se coma el
    espacio que le sobra al contenido)."""
    f, lay = panel(titulo)
    lay.itemAt(0).widget().setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    return f, lay


class BusScada(Bus):
    puente = Signal(bool)
    planta = Signal(dict)
    evento = Signal(dict)
    log = Signal(str)


class Tapita:
    """Una tapita dibujada en el sinoptico."""

    def __init__(self, x, fase):
        self.x = x
        self.fase = fase          # viajando, esperando, inspeccion, saliendo
        self.t_fase = time.monotonic()
        self.color = None
        self.estado = None
        self.pieza_id = None

    def pasar_a(self, fase):
        self.fase = fase
        self.t_fase = time.monotonic()


class Sinoptico(QWidget):
    """Vista lateral de la maqueta: entrada con sensor IR, cinta, tunel con
    camara y LEDs (ESP32-P4 arriba), motor paso a paso y salida."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(760, 400)
        self.planta: dict = {}
        self.puente_ok = False
        self.broker_ok = False
        self.tapitas: list[Tapita] = []
        self.desplazamiento = 0.0   # avance acumulado de la cinta, en px
        self.t_ir = 0.0
        self.t_flash = 0.0
        self._t_anterior = time.monotonic()
        timer = QTimer(self)
        timer.timeout.connect(self._tick)
        timer.start(33)

    # ---- estado ----
    @property
    def comunicacion_ok(self):
        return self.broker_ok and self.puente_ok and self.planta.get("esp32", False)

    def inspeccionando(self):
        return any(t.fase == "inspeccion" for t in self.tapitas)

    def cinta_moviendose(self):
        return (self.comunicacion_ok and self.planta.get("cinta", False)
                and not self.inspeccionando())

    def velocidad(self):
        """Fraccion de cinta por segundo."""
        return 1 / max(TIEMPO_CRUCE_S, 1)

    def velocidad_hasta_camara(self):
        """La tapita tarda el retardo de disparo en ir del sensor a la camara."""
        delay_s = (self.planta.get("delay_ms") or 4200) / 1000
        return (POS_CAMARA - POS_SENSOR) / max(delay_s, 0.5)

    # ---- eventos de la planta ----
    def on_ir(self):
        self.t_ir = time.monotonic()
        self.tapitas.append(Tapita(POS_SENSOR, "viajando"))

    def on_captura(self):
        self.t_flash = time.monotonic()
        t = self._candidata_camara()
        if t is None:
            t = Tapita(POS_CAMARA, "inspeccion")
            self.tapitas.append(t)
        t.x = POS_CAMARA
        t.pasar_a("inspeccion")

    def on_resultado(self, pieza_id, color, estado):
        t = next((t for t in self.tapitas if t.fase == "inspeccion"), None) \
            or self._candidata_camara()
        if t is None:  # foto manual sin deteccion previa
            t = Tapita(POS_CAMARA, "saliendo")
            self.tapitas.append(t)
        t.x = max(t.x, POS_CAMARA)
        t.pieza_id, t.color, t.estado = pieza_id, color, estado
        t.pasar_a("saliendo")

    def on_color(self, pieza_id, color):
        for t in self.tapitas:
            if t.pieza_id == pieza_id:
                t.color = color

    def _candidata_camara(self):
        candidatas = [t for t in self.tapitas if t.fase in ("viajando", "esperando")]
        return max(candidatas, key=lambda t: t.x) if candidatas else None

    def _tick(self):
        ahora = time.monotonic()
        dt = ahora - self._t_anterior
        self._t_anterior = ahora
        if not self.comunicacion_ok:
            self.tapitas.clear()
        elif self.cinta_moviendose():
            v = self.velocidad()
            self.desplazamiento += v * dt * self._largo_px()
            for t in self.tapitas:
                if t.fase == "viajando":
                    t.x += self.velocidad_hasta_camara() * dt
                    if t.x >= POS_CAMARA:
                        t.x = POS_CAMARA
                        t.pasar_a("esperando")
                elif t.fase == "saliendo":
                    t.x += v * dt
                elif t.fase == "esperando" and ahora - t.t_fase > ESPERA_CAPTURA_S:
                    t.pasar_a("saliendo")  # nadie la fotografio: sigue de largo
        for t in self.tapitas:
            if t.fase == "inspeccion" and ahora - t.t_fase > ESPERA_CAPTURA_S * 2:
                t.pasar_a("saliendo")  # la captura fallo y no llego resultado
        self.tapitas = [t for t in self.tapitas if t.x < 1.08]
        self.update()

    # ---- dibujo ----
    def _geom(self):
        w, h = self.width(), self.height()
        x0, x1 = w * 0.08, w * 0.92
        r = max(14.0, h * 0.04)
        y_cinta = h * 0.68
        return w, h, x0, x1, r, y_cinta

    def _largo_px(self):
        _, _, x0, x1, _, _ = self._geom()
        return x1 - x0

    def _px(self, pos):
        _, _, x0, x1, _, _ = self._geom()
        return x0 + pos * (x1 - x0)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h, x0, x1, r, y = self._geom()
        alto_tunel = min(h * 0.45, 260)
        ahora = time.monotonic()
        ok = self.comunicacion_ok
        moviendo = self.cinta_moviendose()

        p.fillRect(self.rect(), QColor("#0e141b"))
        fuente_k = QFont("monospace", 8)
        fuente_txt = QFont("monospace", 9)

        # base y patas
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor("#161e27"))
        base_y = y + 2 * r + 26
        p.drawRect(QRectF(x0 - r - 30, base_y, (x1 - x0) + 2 * r + 60, 14))
        for xs in (x0, x1):
            p.drawRect(QRectF(xs - 4, y + r, 8, base_y - y - r))

        # cinta
        cinta = QPainterPath()
        cinta.addRoundedRect(QRectF(x0 - r, y, (x1 - x0) + 2 * r, 2 * r), r, r)
        p.setBrush(QColor("#2a333d"))
        p.setPen(QPen(QColor(LINEA), 1.5))
        p.drawPath(cinta)
        paso = 22.0
        off = self.desplazamiento % paso
        p.setPen(QPen(QColor("#4a5866" if moviendo else "#36424e"), 2))
        xm = x0 + off
        while xm < x1:
            p.drawLine(QPointF(xm, y + 2), QPointF(xm, y + 6))
            p.drawLine(QPointF(x1 - (xm - x0), y + 2 * r - 6),
                       QPointF(x1 - (xm - x0), y + 2 * r - 2))
            xm += paso

        # rodillos (el de salida es el motriz)
        angulo = self.desplazamiento / r
        for xr in (x0, x1):
            p.setBrush(QColor("#1e2730"))
            p.setPen(QPen(QColor(LINEA), 1.5))
            p.drawEllipse(QPointF(xr, y + r), r * 0.75, r * 0.75)
            p.setPen(QPen(QColor(CIAN if moviendo else GRIS), 2))
            for k in range(2):
                a = angulo + k * math.pi / 2
                p.drawLine(QPointF(xr - math.cos(a) * r * 0.6, y + r - math.sin(a) * r * 0.6),
                           QPointF(xr + math.cos(a) * r * 0.6, y + r + math.sin(a) * r * 0.6))

        # motor paso a paso + driver, al pie del rodillo de salida
        mx, my = x1 + r + 14, y + r - 18
        p.setBrush(QColor("#1e2730"))
        p.setPen(QPen(QColor(CIAN if moviendo else LINEA), 1.5))
        p.drawRect(QRectF(mx, my, 36, 36))
        p.setPen(QPen(QColor(LINEA), 3))
        p.drawLine(QPointF(x1, y + r), QPointF(mx, y + r))
        p.setPen(QColor(GRIS))
        p.setFont(fuente_k)
        p.drawText(QRectF(mx - 40, my + 40, 116, 30), Qt.AlignmentFlag.AlignHCenter,
                   "MOTOR PAP\nTB6600")
        p.setPen(QColor(CIAN if moviendo else GRIS))
        p.drawText(QRectF(mx, my, 36, 36), Qt.AlignmentFlag.AlignCenter, "M")

        # tunel de inspeccion
        tx0, tx1 = self._px(TUNEL[0]), self._px(TUNEL[1])
        techo = y - alto_tunel
        p.setBrush(QColor("#18212b"))
        p.setPen(QPen(QColor(LINEA), 1.5))
        p.drawRect(QRectF(tx0, techo, tx1 - tx0, 12))
        p.drawRect(QRectF(tx0, techo, 10, alto_tunel - 46))
        p.drawRect(QRectF(tx1 - 10, techo, 10, alto_tunel - 46))
        # tiras LED
        p.setPen(QPen(QColor("#fff3c4" if ok else "#3f4b57"), 3))
        p.drawLine(QPointF(tx0 + 16, techo + 16), QPointF(tx0 + 60, techo + 16))
        p.drawLine(QPointF(tx1 - 60, techo + 16), QPointF(tx1 - 16, techo + 16))
        # camara y su campo de vision
        cx = self._px(POS_CAMARA)
        flash = ok and ahora - self.t_flash < 0.35
        campo = QPainterPath()
        campo.moveTo(cx - 8, techo + 32)
        campo.lineTo(cx + 8, techo + 32)
        campo.lineTo(cx + 46, y)
        campo.lineTo(cx - 46, y)
        campo.closeSubpath()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(255, 255, 255, 90) if flash else QColor(79, 195, 247, 22))
        p.drawPath(campo)
        p.setBrush(QColor("#1e2730"))
        p.setPen(QPen(QColor(CIAN if self.inspeccionando() else LINEA), 1.5))
        p.drawRect(QRectF(cx - 16, techo + 14, 32, 18))
        p.setBrush(QColor("#0e141b"))
        p.drawEllipse(QPointF(cx, techo + 32), 6, 4)
        p.setPen(QColor(GRIS))
        p.setFont(fuente_k)
        p.drawText(QRectF(cx + 22, techo + 16, 120, 14), Qt.AlignmentFlag.AlignLeft,
                   "CÁMARA IMX708")

        # sensor IR, adentro del tunel junto a la boca de entrada
        sx = self._px(POS_SENSOR)
        sy = y - alto_tunel * 0.42
        p.setBrush(QColor("#1e2730"))
        p.setPen(QPen(QColor(LINEA), 1.5))
        p.drawRect(QRectF(tx0 + 10, sy - 4, sx - tx0 - 4, 6))  # soporte desde la pared
        ir_activo = ok and ahora - self.t_ir < 0.8
        p.setBrush(QColor("#0f3d2a" if ok else "#232c35"))
        p.drawRect(QRectF(sx - 10, sy - 8, 20, 16))
        p.setPen(QPen(QColor(ROJO if ir_activo else "#5a2a26"), 2, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(sx, sy + 8), QPointF(sx, y - 2))
        p.setPen(QColor(ROJO if ir_activo else GRIS))
        p.drawText(QRectF(sx + 16, sy - 14, 120, 30), Qt.AlignmentFlag.AlignLeft,
                   f"SENSOR IR\n{'DETECCIÓN' if ir_activo else 'FC-51'}")
        # ESP32-P4 sobre el tunel
        ex = (tx0 + tx1) / 2
        p.setBrush(QColor("#10161d"))
        p.setPen(QPen(QColor(VERDE if ok else ROJO), 1.5))
        p.drawRect(QRectF(ex - 60, techo - 30, 120, 26))
        p.setPen(QColor("#d7e0ea"))
        p.setFont(fuente_txt)
        p.drawText(QRectF(ex - 60, techo - 30, 120, 26), Qt.AlignmentFlag.AlignCenter,
                   "ESP32-P4  TinyML")
        p.setBrush(QColor(VERDE if ok else ROJO))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(ex + 52, techo - 24), 3, 3)

        # rotulos de entrada y salida
        p.setPen(QColor(GRIS))
        p.setFont(fuente_k)
        p.drawText(QRectF(x0 - r - 30, y + 2 * r + 6, 120, 16),
                   Qt.AlignmentFlag.AlignLeft, "ENTRADA →")
        p.drawText(QRectF(x1 - 90, y + 2 * r + 6, 120, 16),
                   Qt.AlignmentFlag.AlignLeft, "→ SALIDA")

        # tapitas
        for t in self.tapitas:
            tx = self._px(t.x)
            relleno = QColor(HEX[norm_color(t.color)]) if t.color else QColor("#9aa7b4")
            borde = QColor(COLOR_ESTADO.get(t.estado, "#0e141b"))
            p.setBrush(relleno)
            p.setPen(QPen(borde, 3 if t.estado else 1))
            p.drawRoundedRect(QRectF(tx - 18, y - 15, 36, 15), 3, 3)
            if t.estado:
                p.setPen(QColor(COLOR_ESTADO.get(t.estado, GRIS)))
                p.setFont(QFont("monospace", 8, QFont.Weight.Bold))
                p.drawText(QRectF(tx - 40, y - 32, 80, 14), Qt.AlignmentFlag.AlignCenter,
                           t.estado.upper())

        self._dibujar_cadena(p, w)

        # estado de la linea
        if not self.broker_ok:
            txt, color = "SIN BROKER MQTT", ROJO
        elif not self.puente_ok:
            txt, color = "PUENTE DETENIDO", ROJO
        elif not self.planta.get("esp32"):
            txt, color = "ESP32 DESCONECTADO", ROJO
        elif self.inspeccionando():
            txt, color = "INSPECCIONANDO", CIAN
        elif self.planta.get("cinta"):
            txt, color = "EN MARCHA", VERDE
        else:
            txt, color = "DETENIDA", GRIS
        p.setPen(QColor(color))
        p.setFont(QFont("monospace", 15, QFont.Weight.Bold))
        p.drawText(QRectF(w - 330, 12, 316, 26), Qt.AlignmentFlag.AlignRight, txt)

        if not ok:
            ayuda = ("Revisar que Mosquitto este corriendo en esta PC." if not self.broker_ok
                     else "Abrir ver_imx708.py en una terminal." if not self.puente_ok
                     else "Revisar el cable USB del ESP32: el puente reconecta solo.")
            p.fillRect(QRectF(0, h * 0.30, w, 64), QColor(11, 15, 20, 215))
            p.setPen(QColor(ROJO))
            p.setFont(QFont("monospace", 14, QFont.Weight.Bold))
            p.drawText(QRectF(0, h * 0.30 + 6, w, 26), Qt.AlignmentFlag.AlignCenter, txt)
            p.setPen(QColor("#d7e0ea"))
            p.setFont(fuente_txt)
            p.drawText(QRectF(0, h * 0.30 + 34, w, 20), Qt.AlignmentFlag.AlignCenter, ayuda)

    def _dibujar_cadena(self, p, w):
        """Cadena de comunicacion: cada enlace en verde o rojo."""
        nodos = [("ESP32-P4", self.planta.get("esp32", False) and self.puente_ok),
                 ("ver_imx708", self.puente_ok), ("BROKER", self.broker_ok),
                 ("SCADA", True)]
        enlaces = ["USB", "MQTT", "MQTT"]
        bx, by, bw, bh, sep = 14, 12, 84, 22, 40
        p.setFont(QFont("monospace", 8))
        for i, (nombre, vivo) in enumerate(nodos):
            x = bx + i * (bw + sep)
            p.setBrush(QColor("#131a22"))
            p.setPen(QPen(QColor(VERDE if vivo else ROJO), 1.2))
            p.drawRect(QRectF(x, by, bw, bh))
            p.setPen(QColor("#d7e0ea"))
            p.drawText(QRectF(x, by, bw, bh), Qt.AlignmentFlag.AlignCenter, nombre)
            if i < len(enlaces):
                vivo_enlace = vivo and nodos[i + 1][1]
                p.setPen(QPen(QColor(VERDE if vivo_enlace else ROJO), 2,
                              Qt.PenStyle.SolidLine if vivo_enlace else Qt.PenStyle.DashLine))
                p.drawLine(QPointF(x + bw, by + bh / 2), QPointF(x + bw + sep, by + bh / 2))
                p.setPen(QColor(GRIS))
                p.drawText(QRectF(x + bw, by + bh, sep, 14), Qt.AlignmentFlag.AlignCenter,
                           enlaces[i])


class PanelMandos(QFrame):
    """Botonera del operador. Cada boton manda un comando por MQTT."""

    def __init__(self, mandar):
        super().__init__(objectName="panel")
        self.setFixedWidth(330)
        self.mandar = mandar
        lay = QVBoxLayout(self)
        self.botones: list[QPushButton] = []

        lay.addWidget(QLabel("MODO", objectName="seccion"))
        fila = QHBoxLayout()
        fila.addWidget(self._boton("MARCHA", "marcha", "Arranca la cinta y el disparo automático",
                                   nombre="marcha"))
        fila.addWidget(self._boton("PARADA", "parada", "Frena la cinta y apaga el disparo automático",
                                   nombre="parada"))
        lay.addLayout(fila)
        lay.addSpacing(10)

        lay.addWidget(QLabel("MANUAL", objectName="seccion"))
        grilla = QGridLayout()
        self.ind_cinta = QLabel("—", objectName="valor")
        self.ind_auto = QLabel("—", objectName="valor")
        grilla.addWidget(QLabel("Cinta"), 0, 0)
        grilla.addWidget(self.ind_cinta, 0, 1)
        grilla.addWidget(self._boton("Arrancar", "cinta_on", "Tecla 'g'"), 0, 2)
        grilla.addWidget(self._boton("Parar", "cinta_off", "Tecla 's'"), 0, 3)
        grilla.addWidget(QLabel("Disparo auto"), 1, 0)
        grilla.addWidget(self.ind_auto, 1, 1)
        grilla.addWidget(self._boton("ON", "auto_on", "Sensor IR -> foto sola (tecla 'a')"), 1, 2)
        grilla.addWidget(self._boton("OFF", "auto_off", "Tecla 'a'"), 1, 3)
        lay.addLayout(grilla)
        lay.addWidget(self._boton("Sacar foto e inspeccionar", "foto",
                                  "Inspecciona lo que esté bajo la cámara (tecla 'i')"))
        lay.addSpacing(10)

        lay.addWidget(QLabel("AJUSTES", objectName="seccion"))
        grilla = QGridLayout()
        self.valores = {}
        for i, (clave, nombre, menos, mas, ayuda) in enumerate([
            ("blue_gain", "Ganancia azul", "azul_menos", "azul_mas",
             "Balance de blancos. En la escuela: bajar 2 veces (bb)"),
            ("red_gain", "Ganancia rojo", "rojo_menos", "rojo_mas", "Balance de blancos (r/R)"),
            ("exposicion_us", "Exposición", "exposicion_menos", "exposicion_mas",
             "Exposición fija de la cámara (1/2)"),
            ("delay_ms", "Retardo disparo", "retardo_menos", "retardo_mas",
             "Tiempo del sensor a la cámara (y/Y)"),
        ]):
            etiqueta = QLabel(nombre)
            etiqueta.setToolTip(ayuda)
            grilla.addWidget(etiqueta, i, 0)
            v = QLabel("—", objectName="valor")
            v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.valores[clave] = v
            grilla.addWidget(v, i, 1)
            grilla.addWidget(self._boton("−", menos, ayuda, ancho=34), i, 2)
            grilla.addWidget(self._boton("+", mas, ayuda, ancho=34), i, 3)
        lay.addLayout(grilla)
        lay.addSpacing(10)

        lay.addWidget(QLabel("CONTADOR SENSOR IR", objectName="seccion"))
        self.lbl_detecciones = QLabel("0", objectName="contador")
        lay.addWidget(self.lbl_detecciones)
        lay.addStretch()
        self.habilitar(False)

    def _boton(self, texto, cmd, ayuda, nombre="", ancho=None):
        b = QPushButton(texto, objectName=nombre)
        b.setToolTip(ayuda)
        if ancho:
            b.setFixedWidth(ancho)
        b.clicked.connect(lambda: self.mandar(cmd))
        self.botones.append(b)
        return b

    def habilitar(self, ok):
        for b in self.botones:
            b.setEnabled(ok)

    def mostrar(self, planta):
        def onoff(lbl, v):
            lbl.setText("?" if v is None else ("ON" if v else "OFF"))
            lbl.setStyleSheet(f"color:{GRIS if v is None else VERDE if v else GRIS}")
        onoff(self.ind_cinta, planta.get("cinta"))
        onoff(self.ind_auto, planta.get("auto"))
        fmt = {
            "blue_gain": lambda v: f"{v:.2f}", "red_gain": lambda v: f"{v:.2f}",
            "exposicion_us": lambda v: f"{v / 1000:.1f} ms", "delay_ms": lambda v: f"{v} ms",
        }
        for clave, lbl in self.valores.items():
            v = planta.get(clave)
            lbl.setText("—" if v is None else fmt[clave](v))
        self.lbl_detecciones.setText(str(planta.get("detecciones", 0)))


class Planta(QWidget):
    """Pestana principal: sinoptico, mandos, ultima pieza, totales y alarmas."""

    def __init__(self, bus: BusScada, db, mandar):
        super().__init__()
        self.db = db
        self.mandar_mqtt = mandar
        self.planta: dict = {}
        self.puente_ok = None
        self.broker_ok = None
        self.ultima: dict | None = None
        self.totales = {"sana": 0, "rota": 0, "dudoso": 0}

        root = QVBoxLayout(self)
        arriba = QHBoxLayout()
        f, lay = panel_fijo("Sinóptico de la planta")
        self.sinoptico = Sinoptico()
        lay.addWidget(self.sinoptico)
        arriba.addWidget(f, 1)
        self.mandos = PanelMandos(self.mandar)
        arriba.addWidget(self.mandos)
        root.addLayout(arriba, 3)

        abajo = QHBoxLayout()
        f, lay = panel_fijo("Última pieza")
        fila = QHBoxLayout()
        self.foto = QLabel("sin imagen")
        self.foto.setFixedSize(150, 150)
        self.foto.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.foto.setStyleSheet("background:#0e141b; border:1px solid #24303c; color:#7d8ea0")
        fila.addWidget(self.foto, alignment=Qt.AlignmentFlag.AlignTop)
        datos = QVBoxLayout()
        self.lbl_color = QLabel("—")
        self.lbl_estado = QLabel("—", objectName="estado_linea")
        self.lbl_ia = QLabel("")
        self.lbl_ia.setWordWrap(True)
        for wdg in (self.lbl_color, self.lbl_estado, self.lbl_ia):
            datos.addWidget(wdg)
        datos.addStretch()
        fila.addLayout(datos, 1)
        lay.addLayout(fila)
        abajo.addWidget(f, 2)

        f, lay = panel_fijo("Totales")
        grilla = QGridLayout()
        self.lbl_totales = {}
        for i, (clave, nombre, color) in enumerate([
            ("total", "INSPECCIONADAS", "#d7e0ea"), ("sana", "SANAS", VERDE),
            ("rota", "ROTAS", ROJO), ("dudoso", "DUDOSAS", AMARILLO),
        ]):
            grilla.addWidget(QLabel(nombre, objectName="k"), (i // 2) * 2, i % 2)
            n = QLabel("0", objectName="contador")
            n.setStyleSheet(f"color:{color}")
            grilla.addWidget(n, (i // 2) * 2 + 1, i % 2)
            self.lbl_totales[clave] = n
        lay.addLayout(grilla)
        lay.addStretch()
        abajo.addWidget(f, 2)

        f, lay = panel_fijo("Eventos y alarmas")
        self.eventos = QListWidget()
        lay.addWidget(self.eventos)
        abajo.addWidget(f, 3)
        root.addLayout(abajo, 2)

        bus.puente.connect(self.on_puente)
        bus.conexion.connect(self.on_broker)
        bus.planta.connect(self.on_planta)
        bus.evento.connect(self.on_evento)
        bus.nueva.connect(self.on_nueva)
        bus.imagen.connect(self.on_imagen)
        bus.veredicto.connect(self.on_veredicto)
        bus.color.connect(self.on_color)
        self.cargar_totales()

    # ---- helpers ----
    def evento(self, texto, nivel="info"):
        color = {"alarma": ROJO, "aviso": AMARILLO, "ok": VERDE, "info": "#d7e0ea",
                 "operador": CIAN}[nivel]
        item = QListWidgetItem(f"{datetime.now():%H:%M:%S}  {texto}")
        item.setForeground(QColor(color))
        self.eventos.insertItem(0, item)
        if self.eventos.count() > 300:
            self.eventos.takeItem(self.eventos.count() - 1)

    def mandar(self, cmd, texto=""):
        nombres = {"marcha": "MARCHA", "parada": "PARADA", "cinta_on": "arrancar cinta",
                   "cinta_off": "parar cinta", "auto_on": "disparo auto ON",
                   "auto_off": "disparo auto OFF", "foto": "foto manual"}
        if cmd in nombres:
            self.evento(f"Operador: {nombres[cmd]}", "operador")
        self.mandar_mqtt(cmd, texto)

    def actualizar_habilitado(self):
        s = self.sinoptico
        s.puente_ok, s.broker_ok, s.planta = bool(self.puente_ok), bool(self.broker_ok), self.planta
        self.mandos.habilitar(s.comunicacion_ok)

    def cargar_totales(self):
        for fila in self.db.execute("SELECT estado, COUNT(*) AS n FROM clasificacion GROUP BY estado"):
            if fila["estado"] in self.totales:
                self.totales[fila["estado"]] += fila["n"]
        self.render_totales()

    def render_totales(self):
        for clave, n in self.totales.items():
            self.lbl_totales[clave].setText(str(n))
        self.lbl_totales["total"].setText(str(sum(self.totales.values())))

    def render_ultima(self):
        r = self.ultima
        if not r:
            return
        color = color_final(r)
        self.lbl_color.setText((color or "—").upper())
        self.lbl_color.setStyleSheet(f"color:{HEX[norm_color(color)]}; font-size:18px; font-weight:bold")
        estado = r.get("estado")
        self.lbl_estado.setText((estado or "—").upper())
        self.lbl_estado.setStyleSheet(f"color:{COLOR_ESTADO.get(estado, '#d7e0ea')}")
        if r.get("veredicto_ia"):
            self.lbl_ia.setText(f"IA: {r['veredicto_ia'].upper()} — {r.get('veredicto_ia_razon') or ''}")
        elif estado == "dudoso":
            self.lbl_ia.setText("IA: revisando…")
        else:
            self.lbl_ia.setText("")

    # ---- eventos ----
    def on_broker(self, ok):
        if ok != self.broker_ok:
            self.evento("Broker MQTT conectado" if ok else "Sin conexión al broker MQTT",
                        "ok" if ok else "alarma")
            self.broker_ok = ok
        self.actualizar_habilitado()

    def on_puente(self, ok):
        if ok != self.puente_ok:
            self.evento("Puente serial (ver_imx708) en línea" if ok
                        else "Puente serial (ver_imx708) detenido", "ok" if ok else "alarma")
            self.puente_ok = ok
        self.actualizar_habilitado()

    def on_planta(self, nuevo):
        viejo = self.planta
        if viejo:
            if nuevo.get("esp32") != viejo.get("esp32"):
                self.evento("ESP32 conectado" if nuevo.get("esp32")
                            else "ESP32 DESCONECTADO (cable USB)",
                            "ok" if nuevo.get("esp32") else "alarma")
            if nuevo.get("esp32") and nuevo.get("cinta") != viejo.get("cinta"):
                self.evento("Cinta en marcha" if nuevo.get("cinta") else "Cinta detenida")
            if nuevo.get("auto") is not None and nuevo.get("auto") != viejo.get("auto"):
                self.evento(f"Disparo automático {'ON' if nuevo.get('auto') else 'OFF'}")
        elif nuevo.get("esp32") is False:
            self.evento("ESP32 desconectado", "alarma")
        self.planta = nuevo
        self.mandos.mostrar(nuevo)
        self.actualizar_habilitado()

    def on_evento(self, ev):
        tipo = ev.get("tipo")
        if tipo == "ir":
            self.sinoptico.on_ir()
            self.evento(f"Sensor IR: tapita #{ev.get('n')} en la entrada")
        elif tipo == "captura":
            self.sinoptico.on_captura()
        elif tipo == "reinicio":
            self.evento("ESP32 reiniciado: valores de fábrica (revisar balance de color)", "aviso")

    def on_nueva(self, data):
        r = {k: data.get(k) for k in ("sesion_id", "color", "prob_rota", "pieza_id")}
        r["estado"] = estado_de(r.get("prob_rota"))
        self.ultima = r
        self.foto.setText("sin imagen")
        if r["estado"] in self.totales:
            self.totales[r["estado"]] += 1
            self.render_totales()
        self.sinoptico.on_resultado(r.get("pieza_id"), r.get("color"), r["estado"])
        nivel = {"rota": "alarma", "dudoso": "aviso"}.get(r["estado"], "info")
        self.evento(f"Pieza {r.get('color') or '?'}: {(r['estado'] or '?').upper()}", nivel)
        self.render_ultima()

    def on_imagen(self, sesion, jpeg):
        if self.ultima and self.ultima.get("sesion_id") == sesion:
            pix = QPixmap()
            pix.loadFromData(jpeg)
            self.foto.setPixmap(pix.scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                                           Qt.TransformationMode.SmoothTransformation))

    def on_veredicto(self, _sesion, data):
        veredicto = data.get("veredicto")
        self.evento(f"IA revisó una dudosa: {(veredicto or '?').upper()}",
                    "alarma" if veredicto == "rota" else "info")
        if self.ultima and self.ultima.get("pieza_id") == data.get("pieza_id"):
            self.ultima["veredicto_ia"] = veredicto
            self.ultima["veredicto_ia_razon"] = data.get("razon")
            self.render_ultima()

    def on_color(self, _sesion, data):
        self.sinoptico.on_color(data.get("pieza_id"), data.get("color"))
        if self.ultima and self.ultima.get("pieza_id") == data.get("pieza_id"):
            self.ultima["color_ia"] = data.get("color")
            self.render_ultima()


class Consola(QWidget):
    """Logs del ESP32 en vivo y envio de comandos sueltos (como en ver_imx708)."""

    def __init__(self, bus: BusScada, mandar):
        super().__init__()
        self.mandar = mandar
        lay = QVBoxLayout(self)
        self.texto = QPlainTextEdit(readOnly=True)
        self.texto.setMaximumBlockCount(3000)
        lay.addWidget(self.texto)
        fila = QHBoxLayout()
        self.entrada = QLineEdit(placeholderText="Comando para el ESP32 (ej: bb, p, x) y Enter")
        self.entrada.returnPressed.connect(self.enviar)
        fila.addWidget(self.entrada)
        b = QPushButton("Enviar")
        b.clicked.connect(self.enviar)
        fila.addWidget(b)
        lay.addLayout(fila)
        bus.log.connect(self.texto.appendPlainText)

    def enviar(self):
        txt = self.entrada.text().strip()
        if txt:
            self.texto.appendPlainText(f">>> {txt}")
            self.mandar("raw", txt)
            self.entrada.clear()


class VentanaScada(QMainWindow):
    def __init__(self, bus: BusScada, db, mandar):
        super().__init__()
        self.setWindowTitle("SCADA · Control de calidad de tapitas")
        self.resize(1400, 900)
        central = QWidget()
        root = QVBoxLayout(central)
        self.setCentralWidget(central)

        header = QHBoxLayout()
        header.addWidget(QLabel("SCADA · CONTROL DE CALIDAD DE TAPITAS", objectName="titulo"))
        header.addStretch()
        self.reloj = QLabel()
        header.addWidget(self.reloj)
        root.addLayout(header)
        timer = QTimer(self)
        timer.timeout.connect(lambda: self.reloj.setText(f"{datetime.now():%d/%m/%Y  %H:%M:%S}"))
        timer.start(1000)

        tabs = QTabWidget()
        self.planta = Planta(bus, db, mandar)
        tabs.addTab(self.planta, "Planta")
        tabs.addTab(Ventana(bus, db), "Producción")
        tabs.addTab(Consola(bus, mandar), "Consola ESP32")
        root.addWidget(tabs, 1)


def iniciar_mqtt(bus: BusScada):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="tapitas-scada")

    def on_connect(c, _u, _f, rc, _p):
        bus.conexion.emit(not rc.is_failure)
        c.subscribe([("tapitas/clasificacion", 1), ("tapitas/imagen/+", 1),
                     ("tapitas/veredicto_ia/+", 1), ("tapitas/color_ia/+", 1),
                     (TOPIC_PUENTE, 1), (TOPIC_PLANTA_ESTADO, 1),
                     (TOPIC_PLANTA_EVENTO, 0), (TOPIC_PLANTA_LOG, 0)])

    def on_disconnect(_c, _u, _f, _rc, _p):
        bus.conexion.emit(False)

    def on_message(_c, _u, msg):
        t = msg.topic
        try:
            if t == "tapitas/clasificacion":
                bus.nueva.emit(json.loads(msg.payload))
            elif t.startswith("tapitas/imagen/"):
                bus.imagen.emit(t.split("/", 2)[2], bytes(msg.payload))
            elif t.startswith("tapitas/veredicto_ia/"):
                bus.veredicto.emit(t.split("/", 2)[2], json.loads(msg.payload))
            elif t.startswith("tapitas/color_ia/"):
                bus.color.emit(t.split("/", 2)[2], json.loads(msg.payload))
            elif t == TOPIC_PUENTE:
                bus.puente.emit(msg.payload == b"online")
            elif t == TOPIC_PLANTA_ESTADO:
                bus.planta.emit(json.loads(msg.payload))
            elif t == TOPIC_PLANTA_EVENTO:
                bus.evento.emit(json.loads(msg.payload))
            elif t == TOPIC_PLANTA_LOG:
                bus.log.emit(msg.payload.decode(errors="replace"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[scada] mensaje invalido en {t}: {e}")

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=10)
    client.connect_async(MQTT_HOST, int(os.environ.get("TAPITAS_MQTT_PORT", "1883")), 30)
    client.loop_start()
    return client


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SCADA)
    bus = BusScada()
    db = db_connect()
    cliente = []

    def mandar(cmd, texto=""):
        # Sin broker no se encola nada: un MARCHA viejo no tiene que llegar tarde.
        if cliente and cliente[0].is_connected():
            cliente[0].publish(TOPIC_PLANTA_COMANDO,
                               json.dumps({"cmd": cmd, "texto": texto}), qos=1)

    ventana = VentanaScada(bus, db, mandar)
    ventana.show()
    # MQTT despues de la ventana: los mensajes retenidos llegan apenas conecta.
    client = iniciar_mqtt(bus)
    cliente.append(client)
    code = app.exec()
    client.loop_stop()
    client.disconnect()
    db.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
