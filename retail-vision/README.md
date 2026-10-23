# Retail Vision · prototipo

Prototipo de analítica por video. Usa la cámara de la laptop para:

- Detectar personas y asignarles un **ID de seguimiento**.
- **Reconocerlas cuando salen del cuadro y regresan** (re-identificación por apariencia), para que conserven su mismo ID.
- Contar **entradas y salidas** al cruzar una línea virtual.
- Medir **visitas y tiempo de permanencia** por zona.
- Generar un **mapa de calor** de por dónde se mueve la gente.
- Guardar todo como **eventos** en una base SQLite, lista para pasar a Supabase.

Cuando funcione con la laptop, solo cambias la `fuente` en `config.yaml` por la URL RTSP de las cámaras de la tienda.

---

## 1. Instalación (una sola vez)

Necesitas **Python 3.10 o más reciente**.

**Mac**
```bash
cd retail-vision
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
cd retail-vision
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

La primera vez que corras el programa se descargan solos el modelo de detección `yolo11n.pt` (~6 MB) y el de re-identificación `yolo26s-reid.onnx` (~28 MB).

**¿Ya tenías instalada la versión anterior?** Solo corre de nuevo `pip install -r requirements.txt` para agregar `onnx` y `onnxruntime`.

> **Mac:** si la cámara no abre, ve a *Configuración del Sistema → Privacidad y seguridad → Cámara* y da permiso a Terminal (o a VS Code si corres desde ahí).

---

## 2. Probar

**Primero, sin cámara** (para confirmar que la instalación quedó bien):
```bash
python main.py --fuente ejemplos/prueba.mp4
```
Es un video sintético con dos personas. Al terminar debe decir: 2 personas, 1 entrada y 1 salida.

**Prueba de regresos** (dos personas salen del cuadro y vuelven; una regresa volteada):
```bash
python main.py --fuente ejemplos/prueba_reid.mp4
```
Debe decir **2 personas distintas y 2 regresos reconocidos**. Sin re-identificación saldrían 4 personas.

**Luego, con la cámara de la laptop:**
```bash
python main.py
```

Se abre una ventana con:
- Un recuadro **gris "identificando..."** durante ~1 segundo mientras toma muestras de una persona nueva.
- Un recuadro **verde** con su **ID** y su rastro.
- Un recuadro **cian "regreso 0.81"** cuando reconoce a alguien que ya había estado; el número es la similitud.
- La **línea amarilla** de entrada, rotulada ADENTRO / AFUERA.
- Las **zonas** de color y el panel con conteos en vivo.

Teclas: **h** muestra/oculta el mapa de calor · **q** sale.

Al salir verás un resumen en la terminal, y en `resultados/` quedan el mapa de calor (`calor-*.png`) y la base de eventos.

**Prueba rápida:** camina de un lado a otro frente a la cámara cruzando la línea; debe sumar entradas y salidas. Quédate unos segundos dentro de la Zona A o B; al salir debe sumar una visita.

Si cuenta las entradas como salidas, cambia `invertir_sentido: true` en `zonas.yaml`.

**Prueba de re-identificación:** colócate frente a la cámara hasta que te asigne un ID, sal por completo del cuadro unos segundos y regresa. Debe aparecer tu mismo ID en cian con "regreso". En la terminal verás una línea como `↺ ID 1 regresó tras 6.2 s (similitud 0.74)`.

---

## Cómo funciona la re-identificación

El tracker solo recuerda a una persona unos segundos. Para que conserve su ID al salir y regresar:

1. Cuando aparece alguien, el sistema toma ~3 muestras de su **apariencia corporal** (ropa, colores, complexión, forma del cuerpo) con un modelo entrenado para distinguir personas. Cada muestra es un vector de 512 números, su "huella".
2. Compara esa huella con las de las personas que **ya no están en cuadro**.
3. Si la similitud supera el `umbral` (0 a 1), le devuelve su ID anterior. Si no, le asigna uno nuevo.
4. Mientras la persona sigue en cuadro, se actualiza su huella con nuevas muestras (de frente, de lado, de espaldas), así cada vez la reconoce mejor.
5. Pasados `memoria_min` minutos sin verla, su huella se borra.

**No usa la cara ni el tono de piel.** La cara es un dato biométrico que exige consentimiento expreso y casi no se ve bien en cámaras de techo; el tono de piel distingue mal entre personas. La apariencia del cuerpo completo es lo que usan los sistemas profesionales de retail.

### Calibrar el umbral

Valores medidos con el modelo `yolo26s-reid`: la misma persona da entre **0.47 y 0.89** (incluso de espaldas o parcialmente cortada) y personas distintas dan entre **0.08 y 0.15**.

En `config.yaml`, sección `reid`:
- **Junta a dos personas distintas en un mismo ID** → sube `umbral` (ej. 0.55).
- **No reconoce a alguien que regresa** → bájalo (ej. 0.38).

La similitud de cada regreso aparece en pantalla y en la terminal; úsala de referencia.

**Límites a considerar:**
- Si dos personas visten casi igual (por ejemplo, el **uniforme del personal**), puede confundirlas. Para el personal se resolverá con el checador.
- Si alguien se quita o se pone una chamarra, cambia mucho su apariencia y puede recibir un ID nuevo.
- Funciona dentro de una visita; **no** sirve para reconocer a un cliente en otro día, y así está pensado por privacidad.

---

## 3. Dibujar tus propias zonas

```bash
python definir_zonas.py
```

Toma una foto de la cámara y te deja dibujar encima:

| Tecla | Acción |
| --- | --- |
| clic | agregar punto a la zona |
| `z` | cerrar la zona (pide el nombre en la terminal) |
| `l` | dibujar la línea de entrada (2 clics) |
| `i` | invertir el sentido de la línea |
| `u` | deshacer punto |
| `c` | borrar todo |
| `s` | guardar y salir |

Las coordenadas se guardan normalizadas (0 a 1), así que sirven aunque cambie la resolución.

---

## 4. Ver el reporte

```bash
python reporte.py            # última sesión
python reporte.py --todas    # lista de sesiones
```

Muestra personas detectadas, entradas/salidas, visitas por zona y entradas por hora, y exporta un CSV con todos los eventos.

---

## Otras opciones

```bash
python main.py --fuente video.mp4   # probar con un video grabado
python main.py --fuente 1           # otra cámara (USB)
python main.py --guardar            # graba el video con anotaciones en resultados/
```

### Conectar una cámara de la tienda

En `config.yaml`:
```yaml
fuente: "rtsp://usuario:clave@192.168.1.64:554/Streaming/Channels/101"
punto_referencia: pies
```
La ruta exacta depende de la marca (Hikvision, Dahua, etc.). Con cámaras de techo conviene `punto_referencia: pies`.

---

## Eventos que se registran

Tabla `eventos` en `resultados/eventos.db`:

| tipo | significado | duracion_seg |
| --- | --- | --- |
| `aparece` | una persona nueva entra al cuadro | – |
| `regresa` | una persona conocida volvió al cuadro | segundos que estuvo fuera |
| `entrada` / `salida` | cruzó la línea en ese sentido | – |
| `visita_zona` | estuvo en una zona (columna `zona`) | tiempo en la zona |
| `desaparece` | se dejó de ver | tiempo total en cámara |

---

## Limitaciones de este prototipo

- **Una sola cámara:** por ahora procesa una cámara a la vez. Para seguir a alguien entre varias cámaras se usa el mismo mecanismo de huellas, compartiendo la galería entre ellas; eso falta implementarlo.
- **Velocidad:** en CPU rinde ~5 a 15 cuadros por segundo; suficiente para probar. Una Mac con chip M o una GPU NVIDIA va mucho más rápido.
- **Cliente vs. empleado:** aún no los distingue. Es el siguiente paso: entrenar un clasificador con fotos del uniforme del personal.

## Siguientes pasos

1. Validar detección y conteo con la laptop.
2. Probar con una grabación de las cámaras de la tienda (`--fuente video.mp4`).
3. Conectar por RTSP a una cámara real.
4. Distinguir cliente / empleado.
5. Mandar eventos a Supabase y armar el dashboard en Next.js.
