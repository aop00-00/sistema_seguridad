"""
Dibuja la línea de entrada y las zonas sobre una imagen de la cámara.

Uso:
    python definir_zonas.py                 # toma una foto de la cámara de config.yaml
    python definir_zonas.py --fuente 1
    python definir_zonas.py --imagen foto.jpg

Controles:
    clic izquierdo  agregar punto
    z               cerrar la zona actual (te pide el nombre en la terminal)
    l               dibujar la línea de entrada (2 clics)
    i               invertir el sentido de la línea
    u               deshacer último punto
    c               borrar todo
    s               guardar en zonas.yaml y salir
    q / ESC         salir sin guardar
"""
import argparse
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import yaml

BASE = Path(__file__).resolve().parent
COLORES = [(255, 170, 60), (90, 200, 120), (60, 140, 255), (200, 90, 220), (40, 210, 230)]


def ascii(t):
    return unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode()


def tomar_foto(fuente):
    fuente = int(fuente) if str(fuente).isdigit() else fuente
    cap = cv2.VideoCapture(fuente)
    frame = None
    for _ in range(15):  # deja que la cámara ajuste exposición
        ok, f = cap.read()
        if ok:
            frame = f
    cap.release()
    if frame is None:
        raise SystemExit("No se pudo leer la cámara. Revisa permisos o usa --imagen.")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuente")
    ap.add_argument("--imagen")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(BASE / "config.yaml", encoding="utf-8"))
    ruta_zonas = BASE / cfg.get("archivo_zonas", "zonas.yaml")
    frame = cv2.imread(args.imagen) if args.imagen else tomar_foto(args.fuente or cfg["fuente"])
    h, w = frame.shape[:2]

    datos = yaml.safe_load(open(ruta_zonas, encoding="utf-8")) if ruta_zonas.exists() else {}
    zonas = [{"nombre": z["nombre"], "puntos": [(int(x * w), int(y * h)) for x, y in z["puntos"]]}
             for z in (datos.get("zonas") or [])]
    linea = None
    invertir = False
    if datos.get("linea_entrada"):
        le = datos["linea_entrada"]
        linea = [(int(le["p1"][0] * w), int(le["p1"][1] * h)), (int(le["p2"][0] * w), int(le["p2"][1] * h))]
        invertir = bool(le.get("invertir_sentido", False))

    estado = {"puntos": [], "modo": "zona", "linea_tmp": []}

    def clic(evento, x, y, *_):
        if evento != cv2.EVENT_LBUTTONDOWN:
            return
        if estado["modo"] == "linea":
            estado["linea_tmp"].append((x, y))
        else:
            estado["puntos"].append((x, y))

    cv2.namedWindow("Definir zonas")
    cv2.setMouseCallback("Definir zonas", clic)
    print(__doc__)

    while True:
        img = frame.copy()
        for i, z in enumerate(zonas):
            c = COLORES[i % len(COLORES)]
            cv2.polylines(img, [np.array(z["puntos"], np.int32)], True, c, 2)
            cv2.putText(img, ascii(z["nombre"]), (z["puntos"][0][0] + 6, z["puntos"][0][1] + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        for p in estado["puntos"]:
            cv2.circle(img, p, 5, (0, 0, 255), -1)
        if len(estado["puntos"]) > 1:
            cv2.polylines(img, [np.array(estado["puntos"], np.int32)], False, (0, 0, 255), 1)
        if linea:
            cv2.line(img, linea[0], linea[1], (0, 230, 255), 3)
            cv2.putText(img, "LINEA" + (" (invertida)" if invertir else ""), linea[0],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2)
        for p in estado["linea_tmp"]:
            cv2.circle(img, p, 6, (0, 230, 255), -1)

        modo = "LINEA: 2 clics" if estado["modo"] == "linea" else "ZONA: clic para puntos, z para cerrar"
        cv2.rectangle(img, (0, h - 34), (w, h), (0, 0, 0), -1)
        cv2.putText(img, f"{modo} | l linea  i invertir  u deshacer  c borrar  s guardar  q salir",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Definir zonas", img)

        if estado["modo"] == "linea" and len(estado["linea_tmp"]) == 2:
            linea = estado["linea_tmp"][:]
            estado["linea_tmp"].clear()
            estado["modo"] = "zona"

        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            print("Salida sin guardar.")
            break
        if k == ord("l"):
            estado["modo"], estado["linea_tmp"] = "linea", []
        elif k == ord("i"):
            invertir = not invertir
        elif k == ord("u") and estado["puntos"]:
            estado["puntos"].pop()
        elif k == ord("c"):
            zonas, linea, estado["puntos"] = [], None, []
        elif k == ord("z"):
            if len(estado["puntos"]) < 3:
                print("Una zona necesita al menos 3 puntos.")
                continue
            nombre = input("Nombre de la zona: ").strip() or f"Zona {len(zonas) + 1}"
            zonas.append({"nombre": nombre, "puntos": estado["puntos"][:]})
            estado["puntos"].clear()
        elif k == ord("s"):
            salida = {
                "zonas": [{"nombre": z["nombre"],
                           "puntos": [[round(x / w, 4), round(y / h, 4)] for x, y in z["puntos"]]}
                          for z in zonas],
            }
            if linea:
                salida["linea_entrada"] = {
                    "p1": [round(linea[0][0] / w, 4), round(linea[0][1] / h, 4)],
                    "p2": [round(linea[1][0] / w, 4), round(linea[1][1] / h, 4)],
                    "invertir_sentido": invertir,
                }
            with open(ruta_zonas, "w", encoding="utf-8") as f:
                yaml.safe_dump(salida, f, allow_unicode=True, sort_keys=False)
            print(f"Guardado en {ruta_zonas}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
