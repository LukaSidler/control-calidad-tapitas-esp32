#!/usr/bin/env python3
"""
tapitas_dashboard.py - Panel web para el proyecto de clasificacion de tapitas.

- Sirve una SPA de un solo archivo (dashboard.html) + Chart.js por CDN.
- API REST:
    GET /api/conteos              -> totales por color
    GET /api/historial?page=&page_size=  -> historial paginado (id DESC)
    GET /api/ultima               -> ultima fila
- Imagenes:
    GET /imagen/{id}              -> FileResponse desde imagen_path (anti-traversal)
- Tiempo real:
    WS  /ws                       -> snapshot inicial + push de filas nuevas
                                     (polling liviano de la DB cada ~1.5 s)

Solo lectura sobre ~/tapitas/clasificacion.db. FastAPI + uvicorn + websockets,
todo de apt (sin venv ni Node).
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import sqlite3
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

# --------------------------------------------------------------------------- #
# Configuracion
# --------------------------------------------------------------------------- #
BASE_DIR = Path(os.environ.get("TAPITAS_BASE_DIR", str(Path.home() / "tapitas")))
DB_PATH = BASE_DIR / "clasificacion.db"
IMG_DIR = (BASE_DIR / "imagenes").resolve()
INDEX_HTML = Path(__file__).with_name("dashboard.html")

HOST = os.environ.get("TAPITAS_DASH_HOST", "0.0.0.0")
PORT = int(os.environ.get("TAPITAS_DASH_PORT", "8000"))
POLL_SECS = float(os.environ.get("TAPITAS_DASH_POLL", "1.5"))

# Colores canonicos de tapita (define el orden de los contadores de la UI)
COLORES = ["naranja", "rojo", "amarillo", "azul", "lila", "blanco", "rosa"]

# Color que se muestra y se cuenta: el de la IA (ver tapitas/color_ia en
# tapitas_ingest.py) si ya llego, si no el que midio el ESP32 por HSV.
_COLOR_FINAL = "COALESCE(color_ia, color)"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("tapitas.dash")


# --------------------------------------------------------------------------- #
# Acceso a datos (SOLO LECTURA)
# --------------------------------------------------------------------------- #
def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def borrar_historial() -> int:
    """Borra todas las filas de clasificacion. Las imagenes en disco quedan
    intactas -- solo se limpian los registros de la DB."""
    with sqlite3.connect(DB_PATH, timeout=5) as conn:
        cur = conn.execute("DELETE FROM clasificacion")
        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'clasificacion'")
        conn.commit()
        return cur.rowcount


def _norm(color) -> str:
    return (color or "").strip().lower()


def _db_safe(default_factory):
    """Si la DB todavia no existe / no tiene la tabla, devuelve un default."""
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **kw):
            try:
                return fn(*a, **kw)
            except sqlite3.OperationalError as exc:
                log.warning("DB no disponible en %s(): %s", fn.__name__, exc)
                return default_factory()
        return wrap
    return deco


_COLS = ("id, timestamp, sesion_id, color, hue, saturation, value, "
         "prob_rota, estado, imagen_path, veredicto_ia, veredicto_ia_razon, "
         "color_ia, color_ia_origen")


def _row_to_dict(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["color_esp32"] = d.get("color")
    if d.get("color_ia"):
        d["color"] = d["color_ia"]
    d["color_norm"] = _norm(d.get("color"))
    d["tiene_imagen"] = bool(d.get("imagen_path"))
    d["imagen_url"] = f"/imagen/{d['id']}" if d.get("imagen_path") else None
    return d


@_db_safe(lambda: {"total": 0, "por_color": {c: 0 for c in COLORES}, "otros": {}})
def query_conteos() -> dict:
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT {_COLOR_FINAL} AS color, COUNT(*) AS n FROM clasificacion "
            f"GROUP BY {_COLOR_FINAL}"
        ).fetchall()
    por_color = {c: 0 for c in COLORES}
    otros: dict[str, int] = {}
    total = 0
    for r in rows:
        total += r["n"]
        c = _norm(r["color"])
        if c in por_color:
            por_color[c] += r["n"]
        else:
            otros[c or "(sin color)"] = otros.get(c or "(sin color)", 0) + r["n"]
    return {"total": total, "por_color": por_color, "otros": otros}


@_db_safe(lambda: {"page": 1, "page_size": 25, "total": 0, "total_pages": 1, "rows": []})
def query_historial(page: int, page_size: int) -> dict:
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM clasificacion").fetchone()[0]
        rows = conn.execute(
            f"SELECT {_COLS} FROM clasificacion ORDER BY id DESC LIMIT ? OFFSET ?",
            (page_size, (page - 1) * page_size),
        ).fetchall()
    total_pages = max(1, (total + page_size - 1) // page_size)
    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "rows": [_row_to_dict(r) for r in rows],
    }


@_db_safe(lambda: None)
def query_ultima():
    with _connect() as conn:
        r = conn.execute(
            f"SELECT {_COLS} FROM clasificacion ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return _row_to_dict(r) if r else None


@_db_safe(list)
def query_desde(last_id: int, limit: int = 100) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT {_COLS} FROM clasificacion WHERE id > ? ORDER BY id ASC LIMIT ?",
            (last_id, limit),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


@_db_safe(list)
def query_recientes(limit: int = 20) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, imagen_path, veredicto_ia, veredicto_ia_razon, "
            "color, color_ia, color_ia_origen "
            "FROM clasificacion ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


@_db_safe(lambda: None)
def query_imagen_path(row_id: int):
    with _connect() as conn:
        r = conn.execute(
            "SELECT imagen_path FROM clasificacion WHERE id = ?", (row_id,)
        ).fetchone()
    return r["imagen_path"] if r else None


# --------------------------------------------------------------------------- #
# WebSocket: manager + poller de la DB
# --------------------------------------------------------------------------- #
class Broadcaster:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._last_id = 0
        self._img_cache: dict[int, tuple[str | None, str | None, str | None]] = {}
        self._task: asyncio.Task | None = None

    async def register(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.add(ws)

    async def unregister(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast(self, msg: dict) -> None:
        data = json.dumps(msg, default=str)
        async with self._lock:
            targets = list(self._clients)
        for ws in targets:
            try:
                await ws.send_text(data)
            except Exception:
                await self.unregister(ws)

    async def start(self) -> None:
        recientes = await asyncio.to_thread(query_recientes, 20)
        self._img_cache = {r["id"]: (r["imagen_path"], r["veredicto_ia"], r["color_ia"])
                           for r in recientes}
        self._last_id = max(self._img_cache) if self._img_cache else 0
        self._task = asyncio.create_task(self._loop())
        log.info("poller iniciado (cada %.1fs, ultimo id=%s)", POLL_SECS, self._last_id)

    async def reset(self) -> None:
        """Post-borrado: los ids anteriores ya no existen, se arranca de cero."""
        self._last_id = 0
        self._img_cache = {}
        await self.broadcast({"tipo": "borrado"})

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(POLL_SECS)
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("error en el poller")

    async def _tick(self) -> None:
        # 1) filas nuevas
        nuevas = await asyncio.to_thread(query_desde, self._last_id, 100)
        if nuevas:
            self._last_id = nuevas[-1]["id"]
            for row in nuevas:
                self._img_cache[row["id"]] = (row["imagen_path"], row["veredicto_ia"],
                                              row["color_ia"])
            conteos = await asyncio.to_thread(query_conteos)
            for row in nuevas:
                await self.broadcast({"tipo": "nueva", "fila": row, "conteos": conteos})

        # 2) imagen_path, veredicto_ia o color_ia que llegaron DESPUES (ventana reciente)
        for r in await asyncio.to_thread(query_recientes, 20):
            rid = r["id"]
            actual = (r["imagen_path"], r["veredicto_ia"], r["color_ia"])
            prev = self._img_cache.get(rid)
            if prev is not None and prev != actual:
                self._img_cache[rid] = actual
                ipath, veredicto, color_ia = actual
                fila = {
                    "id": rid,
                    "tiene_imagen": bool(ipath),
                    "imagen_url": f"/imagen/{rid}" if ipath else None,
                    "veredicto_ia": veredicto,
                    "veredicto_ia_razon": r["veredicto_ia_razon"],
                }
                msg = {"tipo": "actualizada", "fila": fila}
                if color_ia != prev[2]:
                    # cambia el color mostrado -> tambien los contadores
                    fila["color"] = color_ia or r["color"]
                    fila["color_norm"] = _norm(fila["color"])
                    fila["color_esp32"] = r["color"]
                    fila["color_ia_origen"] = r["color_ia_origen"]
                    msg["conteos"] = await asyncio.to_thread(query_conteos)
                await self.broadcast(msg)

        # 3) recorte de cache
        if len(self._img_cache) > 200:
            for k in sorted(self._img_cache)[:-200]:
                self._img_cache.pop(k, None)


bcast = Broadcaster()


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not DB_PATH.exists():
        log.warning("la DB no existe aun: %s (se llenara cuando ingest reciba datos)",
                    DB_PATH)
    await bcast.start()
    log.info("dashboard escuchando en http://%s:%s", HOST, PORT)
    try:
        yield
    finally:
        await bcast.stop()
        log.info("dashboard detenido")


app = FastAPI(title="Tapitas Dashboard", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    try:
        return HTMLResponse(INDEX_HTML.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise HTTPException(500, "falta dashboard.html junto al script")


@app.delete("/api/historial")
async def api_borrar_historial():
    n = await asyncio.to_thread(borrar_historial)
    await bcast.reset()
    log.info("historial borrado (%d filas)", n)
    return {"borradas": n}


@app.get("/api/conteos")
async def api_conteos():
    return await asyncio.to_thread(query_conteos)


@app.get("/api/historial")
async def api_historial(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
):
    return await asyncio.to_thread(query_historial, page, page_size)


@app.get("/api/ultima")
async def api_ultima():
    return JSONResponse(await asyncio.to_thread(query_ultima))


@app.get("/imagen/{row_id}")
async def imagen(row_id: int):
    ipath = await asyncio.to_thread(query_imagen_path, row_id)
    if not ipath:
        raise HTTPException(404, "esta tapita no tiene imagen")
    try:
        real = Path(ipath).resolve(strict=True)
    except (FileNotFoundError, RuntimeError):
        raise HTTPException(404, "archivo de imagen no encontrado")
    if real != IMG_DIR and IMG_DIR not in real.parents:
        raise HTTPException(403, "ruta de imagen fuera del directorio permitido")
    return FileResponse(real, media_type="image/jpeg")


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    await bcast.register(websocket)
    try:
        snapshot = {
            "tipo": "snapshot",
            "conteos": await asyncio.to_thread(query_conteos),
            "ultima": await asyncio.to_thread(query_ultima),
            "historial": await asyncio.to_thread(query_historial, 1, 25),
        }
        await websocket.send_text(json.dumps(snapshot, default=str))
        while True:  # el cliente solo manda pings/keepalive; los descartamos
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("error en websocket")
    finally:
        await bcast.unregister(websocket)


def main() -> int:
    uvicorn.run(app, host=HOST, port=PORT, log_level="info", ws="websockets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
