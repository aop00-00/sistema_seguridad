"""
Retail Vision · prototipo de analítica por video
================================================
Detecta personas, les asigna un ID de seguimiento, cuenta entradas y salidas
con una línea virtual, mide el tiempo de permanencia por zona y genera un
mapa de calor. Todo se registra como eventos en SQLite.

Uso:
    python main.py                          # usa config.yaml (cámara de la laptop)
    python main.py --fuente video.mp4       # probar con un video
    python main.py --guardar                # graba el video con anotaciones
    python main.py --sin-ventana --max-frames 300

Teclas en la ventana:
    q / ESC  salir      h  mostrar/ocultar mapa de calor
"""
import argparse
import sqlite3
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import yaml

BASE = Path(__file__).resolve().parent

COLORES_ZONA = [(255, 170, 60), (90, 200, 120), (60, 140, 255), (200, 90, 220), (40, 210, 230)]


# ─────────────────────────── utilidades ───────────────────────────

def ascii(texto: str) -> str:
    """OpenCV no dibuja acentos: los quitamos solo para mostrar en pantalla."""
    return unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()


def cargar_config(ruta_config: Path) -> dict:
    with open(ruta_config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ruta_zonas = BASE / cfg.get("archivo_zonas", "zonas.yaml")
    zonas_cfg = {}
    if ruta_zonas.exists():
        with open(ruta_zonas, encoding="utf-8") as f:
            zonas_cfg = yaml.safe_load(f) or {}
    cfg["zonas"] = zonas_cfg.get("zonas") or []
    cfg["linea_entrada"] = zonas_cfg.get("linea_entrada")
    return cfg


def elegir_dispositivo(preferido: str):
    if str(preferido) != "auto":
        return preferido
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def interpretar_fuente(fuente):
    texto = str(fuente)
    if texto.isdigit():
        return int(texto), False
    es_stream = texto.lower().startswith(("rtsp://", "http://", "https://"))
    return texto, not es_stream  # (fuente, es_archivo)


# ─────────────────────────── registro de eventos ───────────────────────────

class Registro:
    """Guarda cada evento en SQLite. Una 'sesión' = una ejecución del programa."""

    def __init__(self, ruta: Path, inicio: datetime):
        ruta.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(ruta)
        self.con.execute(
            """CREATE TABLE IF NOT EXISTS eventos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sesion TEXT NOT NULL,
                ts TEXT NOT NULL,
                track_id INTEGER,
                tipo TEXT NOT NULL,       -- aparece | entrada | salida | visita_zona | desaparece
                zona TEXT,
                duracion_seg REAL
            )"""
        )
        self.inicio = inicio
        self.sesion = inicio.strftime("%Y%m%d-%H%M%S")
        self._sin_guardar = 0

    def evento(self, t: float, track_id: int, tipo: str, zona=None, duracion=None):
        ts = (self.inicio + timedelta(seconds=t)).isoformat(timespec="seconds")
        dur = round(duracion, 2) if duracion is not None else None
        self.con.execute(
            "INSERT INTO eventos (sesion, ts, track_id, tipo, zona, duracion_seg) VALUES (?,?,?,?,?,?)",
            (self.sesion, ts, track_id, tipo, zona, dur),
        )
        self._sin_guardar += 1
        if self._sin_guardar >= 25:
            self.con.commit()
            self._sin_guardar = 0

    def cerrar(self):
        self.con.commit()
        self.con.close()


# ─────────────────────────── geometría ───────────────────────────

class Zona:
    def __init__(self, nombre, puntos_norm, w, h, color):
        self.nombre = nombre
        self.poly = np.array([[int(x * w), int(y * h)] for x, y in puntos_norm], np.int32)
        self.color = color
        self.visitas = 0
        self.tiempo_total = 0.0
        self.ocupacion = 0

    def contiene(self, pt) -> bool:
        return cv2.pointPolygonTest(self.poly, (float(pt[0]), float(pt[1])), False) >= 0

    def promedio(self) -> float:
        return self.tiempo_total / self.visitas if self.visitas else 0.0


class LineaEntrada:
    """Línea virtual. Usa una franja de tolerancia para no contar 'temblores'."""

    def __init__(self, cfg, w, h):
        self.p1 = np.array([cfg["p1"][0] * w, cfg["p1"][1] * h], float)
        self.p2 = np.array([cfg["p2"][0] * w, cfg["p2"][1] * h], float)
        self.invertir = bool(cfg.get("invertir_sentido", False))
        self.largo = float(np.linalg.norm(self.p2 - self.p1)) or 1.0
        self.margen = 0.025 * float(np.hypot(w, h))
        self.entradas = 0
        self.salidas = 0

    def distancia(self, pt) -> float:
        d = self.p2 - self.p1
        return (d[0] * (pt[1] - self.p1[1]) - d[1] * (pt[0] - self.p1[0])) / self.largo

    def dentro_del_tramo(self, pt) -> bool:
        d = self.p2 - self.p1
        t = np.dot(np.asarray(pt, float) - self.p1, d) / (self.largo ** 2)
        return -0.05 <= t <= 1.05

    def lado(self, pt, lado_anterior: int) -> int:
        dist = self.distancia(pt)
        if dist > self.margen:
            return 1
        if dist < -self.margen:
            return -1
        return lado_anterior  # dentro de la franja: no cambia

    def tipo_cruce(self, lado_anterior: int, lado_nuevo: int):
        if lado_anterior == 0 or lado_nuevo == lado_anterior:
            return None
        es_entrada = (lado_anterior == -1 and lado_nuevo == 1)
        if self.invertir:
            es_entrada = not es_entrada
        return "entrada" if es_entrada else "salida"

    def punto_etiqueta(self, lado_buscado: int):
        """Punto a un costado de la línea, del lado indicado (para rotular)."""
        medio = self.p1 + (self.p2 - self.p1) * 0.12
        d = (self.p2 - self.p1) / self.largo
        normal = np.array([-d[1], d[0]])
        for signo in (1, -1):
            cand = medio + signo * normal * (self.margen + 40)
            if np.sign(self.distancia(cand)) == lado_buscado:
                return int(cand[0]), int(cand[1])
        return int(medio[0]), int(medio[1])


@dataclass
class Persona:
    id: int
    primera_vez: float
    ultima_vez: float
    lado: int = 0
    zonas: dict = field(default_factory=dict)   # nombre -> {"inicio": t, "fuera_desde": t | None}
    rastro: list = field(default_factory=list)


# ─────────────────────────── analítica ───────────────────────────

class Analitica:
    def __init__(self, cfg, w, h, registro: Registro):
        self.cfg = cfg
        self.w, self.h = w, h
        self.reg = registro
        self.zonas = [Zona(z["nombre"], z["puntos"], w, h, COLORES_ZONA[i % len(COLORES_ZONA)])
                      for i, z in enumerate(cfg["zonas"])]
        self.linea = LineaEntrada(cfg["linea_entrada"], w, h) if cfg.get("linea_entrada") else None
        self.personas: dict[int, Persona] = {}
        self.ids_vistos: set[int] = set()
        self.regresos = 0
        self.calor = np.zeros((h // 4 + 1, w // 4 + 1), np.float32)
        self.t_perdida = float(cfg.get("tiempo_perdida_seg", 3.0))
        self.gracia = float(cfg.get("gracia_zona_seg", 0.7))
        self.visita_min = float(cfg.get("visita_minima_seg", 1.0))
        self.usar_pies = cfg.get("punto_referencia", "centro") == "pies"

    def punto(self, caja):
        x1, y1, x2, y2 = caja
        return (int((x1 + x2) / 2), int(y2 if self.usar_pies else (y1 + y2) / 2))

    def actualizar(self, t: float, detecciones):
        """detecciones: lista de (track_id, caja xyxy)."""
        vistos_ahora = set()
        for tid, caja in detecciones:
            pt = self.punto(caja)
            vistos_ahora.add(tid)
            p = self.personas.get(tid)
            if p is None:
                p = Persona(id=tid, primera_vez=t, ultima_vez=t)
                self.personas[tid] = p
                if tid not in self.ids_vistos:
                    self.ids_vistos.add(tid)
                    self.reg.evento(t, tid, "aparece")
            p.ultima_vez = t
            p.rastro.append(pt)
            p.rastro = p.rastro[-40:]

            # Línea de entrada
            if self.linea and self.linea.dentro_del_tramo(pt):
                nuevo = self.linea.lado(pt, p.lado)
                cruce = self.linea.tipo_cruce(p.lado, nuevo)
                if cruce == "entrada":
                    self.linea.entradas += 1
                    self.reg.evento(t, tid, "entrada")
                elif cruce == "salida":
                    self.linea.salidas += 1
                    self.reg.evento(t, tid, "salida")
                p.lado = nuevo

            # Zonas
            for z in self.zonas:
                dentro = z.contiene(pt)
                estado = p.zonas.get(z.nombre)
                if dentro:
                    if estado is None:
                        p.zonas[z.nombre] = {"inicio": t, "fuera_desde": None}
                    else:
                        estado["fuera_desde"] = None
                elif estado is not None:
                    if estado["fuera_desde"] is None:
                        estado["fuera_desde"] = t
                    elif t - estado["fuera_desde"] >= self.gracia:
                        self._cerrar_visita(p, z, estado["fuera_desde"])

            # Mapa de calor
            cv2.circle(self.calor, (pt[0] // 4, pt[1] // 4), 6, 1.0, -1)

        # Personas que dejaron de verse
        for tid in [k for k, p in self.personas.items()
                    if k not in vistos_ahora and t - p.ultima_vez > self.t_perdida]:
            self._despedir(self.personas.pop(tid))

        # Ocupación actual por zona
        for z in self.zonas:
            z.ocupacion = sum(1 for p in self.personas.values()
                              if z.nombre in p.zonas and p.zonas[z.nombre]["fuera_desde"] is None
                              and p.ultima_vez == t)

    def registrar_regreso(self, t: float, pid: int, ausencia: float):
        """La re-identificación reconoció a alguien que ya se había ido."""
        self.regresos += 1
        self.reg.evento(t, pid, "regresa", duracion=ausencia)

    def _cerrar_visita(self, p: Persona, z: Zona, t_fin: float):
        estado = p.zonas.pop(z.nombre)
        dur = t_fin - estado["inicio"]
        if dur >= self.visita_min:
            z.visitas += 1
            z.tiempo_total += dur
            self.reg.evento(t_fin, p.id, "visita_zona", zona=z.nombre, duracion=dur)

    def _despedir(self, p: Persona):
        for z in self.zonas:
            if z.nombre in p.zonas:
                fin = p.zonas[z.nombre]["fuera_desde"] or p.ultima_vez
                self._cerrar_visita(p, z, fin)
        self.reg.evento(p.ultima_vez, p.id, "desaparece", duracion=p.ultima_vez - p.primera_vez)

    def cerrar_todo(self):
        for p in list(self.personas.values()):
            self._despedir(p)
        self.personas.clear()

    # ─── dibujo ───

    def mapa_calor(self, frame):
        if self.calor.max() <= 0:
            return frame
        suave = cv2.GaussianBlur(self.calor, (0, 0), 6)
        norm = (suave / suave.max() * 255).astype(np.uint8)
        norm = cv2.resize(norm, (self.w, self.h))
        color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        mascara = (norm > 8)[..., None]
        mezcla = cv2.addWeighted(frame, 0.45, color, 0.55, 0)
        return np.where(mascara, mezcla, frame)

    def imagen_calor(self, frame):
        """Mapa de calor final sobre una imagen limpia, con zonas y línea."""
        img = self.mapa_calor(frame)
        for z in self.zonas:
            cv2.polylines(img, [z.poly], True, z.color, 2)
            x, y = z.poly[0]
            cv2.putText(img, ascii(z.nombre), (int(x) + 6, int(y) + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, z.color, 2, cv2.LINE_AA)
        if self.linea:
            cv2.line(img, tuple(self.linea.p1.astype(int)), tuple(self.linea.p2.astype(int)),
                     (0, 230, 255), 2, cv2.LINE_AA)
        return img

    def dibujar(self, frame, detecciones, fps, ver_calor, pendientes=(), notas=None):
        notas = notas or {}
        if ver_calor:
            frame = self.mapa_calor(frame)

        capa = frame.copy()
        for z in self.zonas:
            cv2.fillPoly(capa, [z.poly], z.color)
        frame = cv2.addWeighted(capa, 0.15, frame, 0.85, 0)
        for z in self.zonas:
            cv2.polylines(frame, [z.poly], True, z.color, 2)
            x, y = z.poly[0]
            cv2.putText(frame, ascii(z.nombre), (int(x) + 6, int(y) + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, z.color, 2, cv2.LINE_AA)

        if self.linea:
            a, b = tuple(self.linea.p1.astype(int)), tuple(self.linea.p2.astype(int))
            cv2.line(frame, a, b, (0, 230, 255), 3, cv2.LINE_AA)
            lado_adentro = -1 if self.linea.invertir else 1
            for texto, lado in (("ADENTRO", lado_adentro), ("AFUERA", -lado_adentro)):
                px, py = self.linea.punto_etiqueta(lado)
                (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                x0 = px - tw // 2
                cv2.rectangle(frame, (x0 - 6, py - th - 6), (x0 + tw + 6, py + 6), (30, 30, 30), -1)
                cv2.putText(frame, texto, (x0, py), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 230, 255), 2, cv2.LINE_AA)

        # Personas que aún se están identificando (~1 s)
        for caja in pendientes:
            x1, y1, x2, y2 = map(int, caja)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (170, 170, 170), 2)
            cv2.putText(frame, "identificando...", (x1 + 4, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (170, 170, 170), 2, cv2.LINE_AA)

        for tid, caja in detecciones:
            x1, y1, x2, y2 = map(int, caja)
            nota = notas.get(tid)
            color = (255, 200, 0) if nota else (80, 255, 80)   # cian = acaba de ser reconocida
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            etiqueta = f"ID {tid}" + (f"  {nota}" if nota else "")
            (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, etiqueta, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 2, cv2.LINE_AA)
            p = self.personas.get(tid)
            if p and len(p.rastro) > 1:
                cv2.polylines(frame, [np.array(p.rastro, np.int32)], False, (80, 255, 80), 1, cv2.LINE_AA)
            cv2.circle(frame, self.punto(caja), 4, (0, 0, 255), -1)

        lineas = [f"En vista: {len(detecciones) + len(pendientes)}",
                  f"Personas distintas: {len(self.ids_vistos)}   Regresos: {self.regresos}"]
        if self.linea:
            lineas.append(f"Entradas: {self.linea.entradas}   Salidas: {self.linea.salidas}")
        for z in self.zonas:
            lineas.append(f"{ascii(z.nombre)}: {z.ocupacion} ahora | {z.visitas} visitas | prom {z.promedio():.0f}s")
        lineas.append(f"FPS: {fps:.1f}   [h] calor  [q] salir")

        # Panel en la esquina inferior izquierda, para no tapar nombres de zonas
        alto = 26 * len(lineas) + 14
        y0 = self.h - alto - 10
        panel = frame.copy()
        cv2.rectangle(panel, (10, y0), (480, self.h - 10), (0, 0, 0), -1)
        frame = cv2.addWeighted(panel, 0.6, frame, 0.4, 0)
        for i, texto in enumerate(lineas):
            cv2.putText(frame, texto, (22, y0 + 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def resumen(self, duracion: float) -> str:
        filas = [
            "",
            "──────── Resumen de la sesión ────────",
            f"Sesión:               {self.reg.sesion}",
            f"Duración:             {duracion:.0f} s",
            f"Personas distintas:   {len(self.ids_vistos)}",
            f"Regresos reconocidos: {self.regresos}",
        ]
        if self.linea:
            filas.append(f"Entradas / salidas:   {self.linea.entradas} / {self.linea.salidas}")
        for z in self.zonas:
            filas.append(f"{z.nombre:<22}{z.visitas} visitas, promedio {z.promedio():.1f} s")
        return "\n".join(filas)


# ─────────────────────────── programa principal ───────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Prototipo de analítica por video para tienda")
    ap.add_argument("--config", default=str(BASE / "config.yaml"))
    ap.add_argument("--fuente", help="Sobrescribe la fuente de config.yaml (0, video.mp4, rtsp://...)")
    ap.add_argument("--guardar", action="store_true", help="Graba el video anotado en resultados/")
    ap.add_argument("--sin-ventana", action="store_true", help="No muestra ventana (servidor o prueba)")
    ap.add_argument("--max-frames", type=int, default=0, help="Detener tras N cuadros (0 = sin límite)")
    args = ap.parse_args()

    cfg = cargar_config(Path(args.config))
    fuente, es_archivo = interpretar_fuente(args.fuente if args.fuente is not None else cfg["fuente"])
    carpeta = BASE / cfg.get("carpeta_resultados", "resultados")
    carpeta.mkdir(parents=True, exist_ok=True)

    print(f"Abriendo fuente: {fuente}")
    cap = cv2.VideoCapture(fuente)
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    if not ok:
        print("\nNo se pudo abrir la fuente de video.")
        if isinstance(fuente, int):
            print("  • Revisa que ninguna otra app (Zoom, Meet, FaceTime) esté usando la cámara.")
            print("  • En Mac: Configuración del Sistema → Privacidad → Cámara → permite tu Terminal o VS Code.")
            print("  • Prueba con otra cámara:  python main.py --fuente 1")
        else:
            print("  • Revisa la ruta del archivo o la URL RTSP (usuario, clave, IP y puerto).")
        sys.exit(1)

    h, w = frame.shape[:2]
    fps_fuente = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dispositivo = elegir_dispositivo(cfg.get("dispositivo", "auto"))
    print(f"Resolución {w}x{h} · dispositivo de cómputo: {dispositivo}")

    from ultralytics import YOLO  # se importa aquí para que los errores de cámara salgan rápido
    modelo = YOLO(cfg.get("modelo", "yolo11n.pt"))

    inicio = datetime.now()
    registro = Registro(BASE / cfg.get("base_datos", "resultados/eventos.db"), inicio)
    analitica = Analitica(cfg, w, h, registro)

    reid = None
    cfg_reid = cfg.get("reid") or {}
    if cfg_reid.get("activo", False):
        from reidentificacion import Reidentificador
        reid = Reidentificador(cfg_reid)
        print(f"Re-identificación activa · modelo {reid.modelo} · umbral {reid.umbral}")
    notas = {}  # id_persona -> (texto, mostrar_hasta_t)

    escritor = None
    if args.guardar:
        ruta_video = carpeta / f"sesion-{registro.sesion}.mp4"
        escritor = cv2.VideoWriter(str(ruta_video), cv2.VideoWriter_fourcc(*"mp4v"), fps_fuente, (w, h))
        print(f"Grabando video anotado en {ruta_video}")

    mostrar = not args.sin_ventana
    ver_calor = False
    n = 0
    t0 = time.time()
    fps_vis, t_ult = 0.0, time.time()
    ultimo = frame.copy()

    try:
        while ok:
            # Reloj: tiempo real para cámara/RTSP, tiempo del video para archivos
            t = n / fps_fuente if es_archivo else time.time() - t0
            ultimo = frame

            res = modelo.track(frame, persist=True, classes=[0],
                               conf=float(cfg.get("confianza", 0.4)),
                               tracker=cfg.get("tracker", "bytetrack.yaml"),
                               device=dispositivo, verbose=False)[0]
            detecciones = []
            if res.boxes is not None and res.boxes.id is not None:
                ids = res.boxes.id.int().tolist()
                cajas = res.boxes.xyxy.cpu().numpy()
                detecciones = list(zip(ids, cajas))

            # Re-identificación: convierte el ID del tracker en un ID de persona estable
            pendientes = []
            if reid:
                detecciones, pendientes = reid.procesar(t, frame, detecciones)
                for t_ev, pid, tipo, ausencia, sim in reid.eventos:
                    if tipo == "regresa":
                        analitica.registrar_regreso(t_ev, pid, ausencia)
                        notas[pid] = (f"regreso {sim:.2f}", t + 3.0)
                        print(f"  ↺ ID {pid} regresó tras {ausencia:.1f} s (similitud {sim:.2f})")
                notas = {k: v for k, v in notas.items() if v[1] > t}

            analitica.actualizar(t, detecciones)

            ahora = time.time()
            fps_vis = 0.9 * fps_vis + 0.1 * (1.0 / max(ahora - t_ult, 1e-6))
            t_ult = ahora

            if mostrar or escritor:
                anotado = analitica.dibujar(frame, detecciones, fps_vis, ver_calor,
                                            pendientes, {k: v[0] for k, v in notas.items()})
            if escritor:
                escritor.write(anotado)
            if mostrar:
                cv2.imshow("Retail Vision", anotado)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (ord("q"), 27):
                    break
                if tecla == ord("h"):
                    ver_calor = not ver_calor

            n += 1
            if args.max_frames and n >= args.max_frames:
                break
            ok, frame = cap.read()
    except KeyboardInterrupt:
        pass
    finally:
        duracion = (n / fps_fuente) if es_archivo else (time.time() - t0)
        analitica.cerrar_todo()
        registro.cerrar()
        cap.release()
        if escritor:
            escritor.release()
        if mostrar:
            cv2.destroyAllWindows()

        ruta_calor = carpeta / f"calor-{registro.sesion}.png"
        cv2.imwrite(str(ruta_calor), analitica.imagen_calor(ultimo))
        print(analitica.resumen(duracion))
        print(f"\nMapa de calor:  {ruta_calor}")
        print(f"Eventos:        {BASE / cfg.get('base_datos', 'resultados/eventos.db')}")
        print("Reporte:        python reporte.py")


if __name__ == "__main__":
    main()
