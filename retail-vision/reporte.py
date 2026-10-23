"""
Resume los eventos registrados y los exporta a CSV.

Uso:
    python reporte.py                 # última sesión
    python reporte.py --sesion 20261020-113000
    python reporte.py --todas         # lista las sesiones guardadas
"""
import argparse
import csv
import sqlite3
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sesion")
    ap.add_argument("--todas", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(BASE / "config.yaml", encoding="utf-8"))
    ruta_db = BASE / cfg.get("base_datos", "resultados/eventos.db")
    if not ruta_db.exists():
        raise SystemExit("Aún no hay eventos. Corre primero: python main.py")
    con = sqlite3.connect(ruta_db)

    if args.todas:
        for sesion, n, ini, fin in con.execute(
                "SELECT sesion, COUNT(*), MIN(ts), MAX(ts) FROM eventos GROUP BY sesion ORDER BY sesion"):
            print(f"{sesion}   {n:>5} eventos   {ini} → {fin}")
        return

    sesion = args.sesion or con.execute("SELECT MAX(sesion) FROM eventos").fetchone()[0]
    if not sesion:
        raise SystemExit("No hay sesiones registradas.")

    def uno(sql, *p):
        return con.execute(sql, (sesion, *p)).fetchone()[0]

    personas = uno("SELECT COUNT(*) FROM eventos WHERE sesion=? AND tipo='aparece'")
    entradas = uno("SELECT COUNT(*) FROM eventos WHERE sesion=? AND tipo='entrada'")
    salidas = uno("SELECT COUNT(*) FROM eventos WHERE sesion=? AND tipo='salida'")
    en_vista = uno("SELECT AVG(duracion_seg) FROM eventos WHERE sesion=? AND tipo='desaparece'") or 0
    regresos = uno("SELECT COUNT(*) FROM eventos WHERE sesion=? AND tipo='regresa'")
    ausencia = uno("SELECT AVG(duracion_seg) FROM eventos WHERE sesion=? AND tipo='regresa'") or 0

    print(f"\nSesión {sesion}")
    print("─" * 46)
    print(f"Personas distintas detectadas:  {personas}")
    print(f"Entradas / salidas por línea:   {entradas} / {salidas}")
    print(f"Tiempo promedio en cámara:      {en_vista:.1f} s")
    print(f"Regresos reconocidos:           {regresos}" + (f"  (ausencia prom. {ausencia:.1f} s)" if regresos else ""))

    zonas = con.execute(
        """SELECT zona, COUNT(*), AVG(duracion_seg), MAX(duracion_seg)
           FROM eventos WHERE sesion=? AND tipo='visita_zona'
           GROUP BY zona ORDER BY COUNT(*) DESC""", (sesion,)).fetchall()
    if zonas:
        print("\nZonas")
        print(f"  {'Zona':<20}{'Visitas':>8}{'Prom (s)':>10}{'Máx (s)':>10}")
        for zona, n, prom, mx in zonas:
            print(f"  {zona:<20}{n:>8}{prom:>10.1f}{mx:>10.1f}")

    por_hora = con.execute(
        """SELECT substr(ts, 12, 2) AS hora, COUNT(*) FROM eventos
           WHERE sesion=? AND tipo='entrada' GROUP BY hora ORDER BY hora""", (sesion,)).fetchall()
    if por_hora:
        print("\nEntradas por hora")
        for hora, n in por_hora:
            print(f"  {hora}:00  {'█' * min(n, 40)} {n}")

    ruta_csv = ruta_db.parent / f"eventos-{sesion}.csv"
    filas = con.execute(
        "SELECT ts, track_id, tipo, zona, duracion_seg FROM eventos WHERE sesion=? ORDER BY id", (sesion,))
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ts", "track_id", "tipo", "zona", "duracion_seg"])
        w.writerows(filas)
    print(f"\nCSV exportado: {ruta_csv}\n")


if __name__ == "__main__":
    main()
