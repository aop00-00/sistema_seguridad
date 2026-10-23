"""
Re-identificación por apariencia
================================
El tracker (ByteTrack) solo recuerda a una persona unos segundos: si sale del
cuadro o queda tapada, al volver le asigna un ID nuevo.

Este módulo le da a cada persona un ID estable. Guarda una "huella" de su
apariencia (ropa, colores, complexión, forma del cuerpo) y, cuando aparece un
track nuevo, la compara contra las personas que ya no están en cuadro. Si se
parece lo suficiente, recupera su ID anterior.

No usa reconocimiento facial: el modelo mira el cuerpo completo, y las huellas
se olvidan después de `memoria_min` minutos sin ver a la persona.
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Identidad:
    id: int
    muestras: deque
    creada: float
    ultima_vez: float
    ultima_muestra: float = -1e9
    regresos: int = 0


@dataclass
class Pendiente:
    """Track nuevo que aún no sabemos si es alguien conocido."""
    inicio: float
    buenas: list = field(default_factory=list)   # muestras con la persona completa en cuadro
    todas: list = field(default_factory=list)


class Reidentificador:
    def __init__(self, cfg: dict):
        from ultralytics.trackers.utils.reid import ReID

        self.modelo = cfg.get("modelo", "yolo26s-reid.onnx")
        self.enc = ReID(self.modelo, device="cpu")
        self.umbral = float(cfg.get("umbral", 0.45))
        self.margen = float(cfg.get("margen", 0.05))
        self.memoria = float(cfg.get("memoria_min", 30)) * 60
        self.n_confirmar = int(cfg.get("muestras_confirmacion", 3))
        self.espera_max = float(cfg.get("espera_max_seg", 1.0))
        self.intervalo = float(cfg.get("intervalo_muestra_seg", 0.5))
        self.max_muestras = int(cfg.get("max_muestras", 40))

        self.identidades: dict[int, Identidad] = {}
        self.track_a_id: dict[int, int] = {}       # ID del tracker → ID de persona
        self.track_visto: dict[int, float] = {}
        self.pendientes: dict[int, Pendiente] = {}
        self._siguiente = 1
        self.eventos: list[tuple] = []             # (t, id_persona, tipo, ausencia_seg, similitud)

    # ─── utilidades ───

    def _embeber(self, frame, cajas):
        xywh = np.array([[(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1] for x1, y1, x2, y2 in cajas], float)
        salida = []
        for v in self.enc(frame, xywh):
            if v is None:
                salida.append(None)
                continue
            v = np.asarray(v, np.float32).ravel()
            n = float(np.linalg.norm(v))
            salida.append(v / n if n > 0 else None)
        return salida

    @staticmethod
    def _completa(caja, w, h) -> bool:
        """La persona se ve bien: no es diminuta ni está cortada por un costado."""
        x1, y1, x2, y2 = caja
        alto = y2 - y1
        if alto < max(48, 0.12 * h):
            return False
        return x1 > 3 and x2 < w - 3

    @staticmethod
    def _puntaje(q, ident: Identidad) -> float:
        """Promedio de las 3 muestras más parecidas de la galería."""
        sims = np.stack(ident.muestras) @ q
        k = min(3, len(sims))
        return float(np.sort(sims)[-k:].mean())

    def _nueva(self, t) -> Identidad:
        ident = Identidad(id=self._siguiente, muestras=deque(maxlen=self.max_muestras), creada=t, ultima_vez=t)
        self.identidades[ident.id] = ident
        self._siguiente += 1
        return ident

    # ─── ciclo por cuadro ───

    def procesar(self, t: float, frame, detecciones):
        """
        detecciones: [(id_tracker, caja_xyxy)]
        Devuelve (confirmadas, pendientes):
          confirmadas = [(id_persona, caja)]   ya identificadas
          pendientes  = [caja]                 aún reuniendo muestras (~1 s)
        """
        h, w = frame.shape[:2]
        self.eventos = []
        presentes = {tid for tid, _ in detecciones}
        for tid in presentes:
            self.track_visto[tid] = t

        # 1. ¿Quién necesita huella en este cuadro?
        pedir = []
        for tid, caja in detecciones:
            pid = self.track_a_id.get(tid)
            if pid is not None and pid not in self.identidades:
                self.track_a_id.pop(tid)  # la identidad expiró
                pid = None
            if pid is None:
                pedir.append((tid, caja))
            elif t - self.identidades[pid].ultima_muestra >= self.intervalo and self._completa(caja, w, h):
                pedir.append((tid, caja))
        embs = {}
        if pedir:
            embs = dict(zip([p[0] for p in pedir], self._embeber(frame, [p[1] for p in pedir])))

        # 2. Personas ya identificadas: actualizar su galería
        ocupadas = set()
        for tid, _ in detecciones:
            pid = self.track_a_id.get(tid)
            if pid is None:
                continue
            ident = self.identidades[pid]
            ident.ultima_vez = t
            ocupadas.add(pid)
            v = embs.get(tid)
            if v is not None:
                ident.muestras.append(v)
                ident.ultima_muestra = t

        # 3. Tracks nuevos: reunir muestras
        listos = []
        for tid, caja in detecciones:
            if tid in self.track_a_id:
                continue
            p = self.pendientes.setdefault(tid, Pendiente(inicio=t))
            v = embs.get(tid)
            if v is not None:
                p.todas.append(v)
                if self._completa(caja, w, h):
                    p.buenas.append(v)
            if len(p.buenas) >= self.n_confirmar or (t - p.inicio >= self.espera_max and p.todas):
                listos.append(tid)

        # 4. Resolver: ¿es alguien que se fue o una persona nueva?
        candidatas = [i for i in self.identidades.values() if i.id not in ocupadas]
        propuestas = []
        for tid in listos:
            p = self.pendientes[tid]
            muestras = p.buenas or p.todas
            q = np.mean(muestras, axis=0)
            q /= np.linalg.norm(q)
            puntajes = sorted(((self._puntaje(q, i), i.id) for i in candidatas), reverse=True)
            propuestas.append((tid, muestras, puntajes))
        propuestas.sort(key=lambda x: x[2][0][0] if x[2] else -1.0, reverse=True)

        usadas = set()
        for tid, muestras, puntajes in propuestas:
            puntajes = [(s, i) for s, i in puntajes if i not in usadas]
            mejor = puntajes[0] if puntajes else None
            segundo = puntajes[1][0] if len(puntajes) > 1 else -1.0
            if mejor and mejor[0] >= self.umbral and mejor[0] - segundo >= self.margen:
                ident = self.identidades[mejor[1]]
                ident.regresos += 1
                self.eventos.append((t, ident.id, "regresa", t - ident.ultima_vez, mejor[0]))
            else:
                ident = self._nueva(t)
                self.eventos.append((t, ident.id, "nueva", None, mejor[0] if mejor else None))
            for v in muestras:
                ident.muestras.append(v)
            ident.ultima_vez = ident.ultima_muestra = t
            usadas.add(ident.id)
            ocupadas.add(ident.id)
            self.track_a_id[tid] = ident.id
            del self.pendientes[tid]

        # 5. Limpieza
        for tid in [k for k in self.pendientes if k not in presentes and t - self.track_visto.get(k, t) > 2.0]:
            del self.pendientes[tid]
        for tid in [k for k, visto in self.track_visto.items() if t - visto > 10.0]:
            self.track_visto.pop(tid)
            self.track_a_id.pop(tid, None)
            self.pendientes.pop(tid, None)
        for pid in [k for k, i in self.identidades.items() if t - i.ultima_vez > self.memoria]:
            del self.identidades[pid]  # privacidad: la huella no se guarda indefinidamente

        # 6. Salida (sin repetir un mismo ID de persona en el cuadro)
        confirmadas, vistos = [], set()
        pendientes = []
        for tid, caja in detecciones:
            pid = self.track_a_id.get(tid)
            if pid is None:
                pendientes.append(caja)
            elif pid not in vistos:
                vistos.add(pid)
                confirmadas.append((pid, caja))
        return confirmadas, pendientes
