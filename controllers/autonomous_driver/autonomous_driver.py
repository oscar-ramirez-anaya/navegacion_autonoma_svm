"""
===============================================================================
  Actividad 3.1 — Deteccion de Peatones con SVM (MR4010.10)
  Controlador Webots: PID Lane Following + LiDAR + SVM
===============================================================================

  Descripcion:
      Controlador autonomo para simulacion en Webots que combina:
        1. Seguimiento de carril con PID (Canny + Hough + EMA)
        2. Deteccion de obstaculos con LiDAR (Sick LMS 291)
        3. Clasificacion peaton/barril con HOG + SVM via sliding window

  Comportamiento:
      - El vehiculo sigue el carril a velocidad crucero (~30 km/h)
      - Cuando el LiDAR detecta un obstaculo a <20m en el rango central,
        se activa la clasificacion por camara
      - Si es peaton: frenado de emergencia (sin intermitentes)
      - Si es barril: frenado de emergencia + hazard flashers
      - Cuando el obstaculo desaparece, se reanuda la marcha

  Pipeline de vision (lane following):
      Camara -> Grises -> Canny(50,150) -> ROI trapezoidal -> Hough -> Error -> EMA -> PID

  Pipeline de deteccion:
      LiDAR -> Obstaculo cercano? -> Camara -> Sliding Window -> HOG -> SVM -> Accion

  Ganancias PID:  Kp=0.008 | Ki=0.0 | Kd=0.015
  Suavizado EMA:  alpha=0.6

  Equipo:
      Antonio Olvera Donlucas          A01795617
      Carlos Monir Radovich Saad       A01797569
      Andres Roberto Osuna Gonzalez    A01796264
      Oscar Alberto Ramirez Anaya      A01795438

  Institucion:
      Instituto Tecnologico y de Estudios Superiores de Monterrey
      Maestria en Inteligencia Artificial

  Fecha: Mayo 2026
===============================================================================
"""

import os
import sys
import math
import numpy as np
import cv2
import traceback

# Intentar cargar joblib y skimage (necesarios para el SVM)
try:
    import joblib
    from skimage.feature import hog
    SVM_AVAILABLE = True
except ImportError:
    print("[WARN] joblib o skimage no disponibles. Clasificacion SVM desactivada.")
    SVM_AVAILABLE = False

# Imports de Webots
from controller import Display, Keyboard
from vehicle import Driver


# ============================================================
# 1. CONSTANTES
# ============================================================

# Velocidad y limites
TARGET_SPEED = 30           # km/h — velocidad crucero (reducida por obstaculos)
MAX_SPEED = 250             # km/h — limite de velocidad
MAX_ANGLE = 0.5             # radianes — angulo maximo del volante

# Ganancias del control PID (mantienen al carro en el centro del carril)
KP = 0.008                  # Proporcional: que tan fuerte gira si se sale de la linea
KI = 0.0                    # Integral: corrige errores acumulados (aqui no lo usamos)
KD = 0.015                  # Derivativo: suaviza el giro para que no "vibre"

# Configuracion del LiDAR
LIDAR_HALF_AREA = 20        # indices a cada lado del centro que revisamos
LIDAR_MAX_DIST = 20.0       # metros — el LiDAR ignora cualquier cosa mas alla de esto

# Estados del vehiculo
STATE_NORMAL = "NORMAL"
STATE_EMERGENCY_PED = "EMERGENCY_PEDESTRIAN"  # peaton detectado
STATE_EMERGENCY_OBJ = "EMERGENCY_OBJECT"       # objeto detectado (barril, caja, cono, etc.)

# Parametros HOG (deben coincidir con el entrenamiento)
HOG_ORIENTATIONS = 11
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

# Ruta del modelo SVM
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
SVM_MODEL_PATH = os.path.join(PROJECT_ROOT, "svm_training", "model", "svm_pedestrian_model.joblib")

# Ruta alternativa: modelo en la misma carpeta del controlador
SVM_MODEL_PATH_LOCAL = os.path.join(SCRIPT_DIR, "svm_pedestrian_model.joblib")

# Debug
DEBUG_EVERY = 30            # imprimir info cada N pasos


# ============================================================
# 2. SEGUIMIENTO DE LINEAS (pipeline Canny + Hough + PID)
# ============================================================

def get_image(camera):
    """Extrae la imagen de la camara y la convierte a una matriz de Numpy (BGRA)."""
    raw = camera.getImage()
    if raw is None:
        return None
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )


def procesar_lineas(image):
    """
    Convierte a escala de grises, aplica Canny para detectar bordes,
    recorta la region de interes (ROI) de la carretera y busca lineas
    con la transformada de Hough.

    Retorna la imagen en grises y las lineas detectadas.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Desenfoque gaussiano para reducir ruido antes de Canny
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Deteccion de bordes con umbrales 50/150
    # (valores tipicos 80/200 detectan poco en imagenes pequenas)
    edges = cv2.Canny(blur, 50, 150)

    # ROI trapezoidal: la zona de la carretera esta en la mitad inferior
    h, w = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, h),                         # esquina inferior izquierda
        (int(0.2 * w), int(0.5 * h)),   # superior izquierda
        (int(0.8 * w), int(0.5 * h)),   # superior derecha
        (w, h),                         # esquina inferior derecha
    ]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Transformada de Hough probabilistica para detectar segmentos de linea
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=15,       # votos minimos para considerar una linea
        minLineLength=8,    # largo minimo en pixeles
        maxLineGap=5,       # maximo hueco permitido
    )
    return gray, lines


def calcular_error_direccion(lines, setpoint):
    """
    Calcula que tan lejos esta el centro del carril respecto al centro
    de la camara (setpoint).
    Filtra lineas casi horizontales (no son bordes de carril) y selecciona
    la linea mas cercana al centro.

    Retorna None si no hay lineas validas.
    """
    if lines is None:
        return None
    candidates = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Descartar lineas horizontales (dx > 3*dy)
        if abs(x2 - x1) > 3 * abs(y2 - y1):
            continue
        mid_x = (x1 + x2) / 2.0
        candidates.append(mid_x - setpoint)

    if not candidates:
        return None
    # Tomar la linea con error mas cercano a cero
    return min(candidates, key=abs)


# ============================================================
# 3. LIDAR Y SVM
# ============================================================

def process_lidar(lidar):
    """
    Procesa los datos del LiDAR Sick LMS 291.
    Solo revisa la region central (±LIDAR_HALF_AREA indices alrededor del centro).
    Devuelve el angulo promedio y la distancia promedio del obstaculo mas cercano.

    Retorna:
        (angulo, distancia) o (None, None) si no hay obstaculos.
    """
    range_data = lidar.getRangeImage()
    if not range_data:
        return None, None

    n = len(range_data)
    center = n // 2
    sumx = 0
    collision_count = 0
    obstacle_dist = 0.0

    # Revisar solo la zona central del LiDAR
    for x in range(center - LIDAR_HALF_AREA, center + LIDAR_HALF_AREA):
        r = range_data[x]
        # Solo considerar lecturas validas dentro del rango maximo
        if r <= LIDAR_MAX_DIST and not math.isinf(r) and not math.isnan(r):
            sumx += x
            collision_count += 1
            obstacle_dist += r

    if collision_count == 0:
        return None, None

    # Calcular angulo promedio (normalizado al FOV) y distancia promedio
    avg_angle = (sumx / collision_count / n - 0.5) * lidar.getFov()
    avg_dist = obstacle_dist / collision_count
    return avg_angle, avg_dist


def detect_pedestrian_svm(bgra_image, model, lidar_dist):
    """
    Busca peatones en la imagen usando sliding window + HOG + SVM.
    Utiliza la distancia del LiDAR para adaptar el tamano de la ventana:
      - Objeto lejos (>14m): ventana chica (32x32)
      - Objeto medio (8-14m): ventana mediana (48x48)
      - Objeto cerca (<8m): ventana grande (64x64)

    Cada ventana se redimensiona a 64x64, se extrae HOG y se clasifica.
    Si la confianza del SVM (decision_function) supera 0.55, es peaton.

    Retorna True si se detecta al menos un peaton.
    """
    h, w = bgra_image.shape[:2]

    # ROI: zona central de la imagen donde aparecen los obstaculos
    roi_start = int(h * 0.25)
    roi_end = int(h * 0.85)
    roi = bgra_image[roi_start:roi_end, :, :3]  # quitar canal alpha
    roi_h, roi_w = roi.shape[:2]

    # Adaptar tamano de ventana segun la distancia del LiDAR
    # Objetos lejos se ven mas chicos en la imagen
    if lidar_dist > 14.0:
        win_size, step = 32, 20   # objeto lejos = ventana chica
    elif lidar_dist > 8.0:
        win_size, step = 48, 24   # objeto medio = ventana mediana
    else:
        win_size, step = 64, 32   # objeto cerca = ventana grande

    # Sliding window 2D (horizontal y vertical)
    for y in range(0, roi_h - win_size + 1, step):
        for x in range(0, roi_w - win_size + 1, step):
            window = roi[y:y + win_size, x:x + win_size]

            # Redimensionar a 64x64 (tamano del entrenamiento)
            window_resized = cv2.resize(window, (64, 64))
            gray = cv2.cvtColor(window_resized, cv2.COLOR_BGR2GRAY)

            # Extraer descriptor HOG
            hog_features = hog(
                gray,
                orientations=HOG_ORIENTATIONS,
                pixels_per_cell=HOG_PIXELS_PER_CELL,
                cells_per_block=HOG_CELLS_PER_BLOCK,
                visualize=False,
                feature_vector=True,
            )

            # Clasificar con SVM: decision_function > 0.55 = peaton
            confidence = model.decision_function([hog_features])[0]
            if confidence > 0.55:
                return True

    # Ninguna ventana detecto peaton — se asume que es barril
    return False


# ============================================================
# 4. CONTROLES DEL VEHICULO
# ============================================================

SLOW_SPEED = 10      # km/h — velocidad reducida al acercarse a un obstaculo
BRAKE_DIST = 8.0     # metros — distancia a la que frena completamente

def aplicar_comandos_vehiculo(driver, estado, steering, target_speed, lidar_dist):
    """
    Aplica los comandos al vehiculo dependiendo del estado actual.
    - NORMAL: sigue el carril a velocidad crucero.
    - PEATON: frena completamente si < 8m, sino reduce a 10 km/h.
    - OBJETO (barril/caja/cono): frena + hazard flashers si < 8m, sino reduce velocidad.
    """
    if estado == STATE_NORMAL:
        driver.setSteeringAngle(steering)
        driver.setCruisingSpeed(target_speed)
        driver.setHazardFlashers(False)
        try:
            driver.setBrakeIntensity(0.0)
        except Exception:
            pass
    elif estado == STATE_EMERGENCY_PED:
        # Peaton detectado
        driver.setSteeringAngle(steering)
        driver.setHazardFlashers(False)
        if lidar_dist is not None and lidar_dist < BRAKE_DIST:
            # Distancia critica: freno total
            driver.setCruisingSpeed(0)
            try:
                driver.setBrakeIntensity(1.0)
            except Exception:
                pass
        else:
            # Lejos todavia: reducir velocidad
            driver.setCruisingSpeed(SLOW_SPEED)
            try:
                driver.setBrakeIntensity(0.0)
            except Exception:
                pass
    elif estado == STATE_EMERGENCY_OBJ:
        # Objeto detectado (barril, caja, cono, etc.)
        driver.setSteeringAngle(steering)
        driver.setHazardFlashers(True)  # intermitentes para objetos
        if lidar_dist is not None and lidar_dist < BRAKE_DIST:
            # Distancia critica: freno total
            driver.setCruisingSpeed(0)
            try:
                driver.setBrakeIntensity(1.0)
            except Exception:
                pass
        else:
            # Lejos todavia: reducir velocidad
            driver.setCruisingSpeed(SLOW_SPEED)
            try:
                driver.setBrakeIntensity(0.0)
            except Exception:
                pass


# ============================================================
# 5. MAIN — BUCLE PRINCIPAL DEL CONTROLADOR
# ============================================================

def main():
    # --- Cargar modelo SVM ---
    svm_model = None
    if SVM_AVAILABLE:
        # Intentar primero la ruta del proyecto, luego la local
        for path in [SVM_MODEL_PATH, SVM_MODEL_PATH_LOCAL]:
            if os.path.exists(path):
                svm_model = joblib.load(path)
                print(f"[INFO] Modelo SVM cargado desde: {path}")
                break
        if svm_model is None:
            print(f"[WARN] Modelo SVM no encontrado.")
            print(f"[WARN] Buscado en: {SVM_MODEL_PATH}")
            print(f"[WARN]          y: {SVM_MODEL_PATH_LOCAL}")
            print("[WARN] Ejecuta primero train_pedestrian_svm.py o copia el .joblib aqui")
            print("[WARN] Sin SVM, todos los obstaculos se trataran como barriles.")

    # --- Inicializacion de Webots ---
    driver = Driver()
    timestep = int(driver.getBasicTimeStep())

    # Camara
    camera = driver.getDevice("camera")
    camera.enable(timestep)
    cam_width = camera.getWidth()
    cam_height = camera.getHeight()
    setpoint = cam_width / 2.0
    print(f"[INFO] Camara {cam_width}x{cam_height} -> setpoint = {setpoint}")

    # LiDAR (Sick LMS 291)
    lidar = driver.getDevice("Sick LMS 291")
    lidar.enable(timestep)
    lidar.enablePointCloud()

    # Teclado para controles manuales
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # --- Variables de control ---
    speed = TARGET_SPEED
    vehicle_state = STATE_NORMAL
    prev_error = 0.0
    integral = 0.0
    smoothed_error = None
    step_count = 0
    svm_check_cnt = 0

    print(f"[INFO] PID: Kp={KP} Ki={KI} Kd={KD}")
    print(f"[INFO] Velocidad crucero: {TARGET_SPEED} km/h")
    print(f"[INFO] LiDAR: max_dist={LIDAR_MAX_DIST}m, half_area={LIDAR_HALF_AREA}")
    print("[INFO] Simulacion iniciada correctamente.")

    # --------------------------------------------------------
    # CICLO PRINCIPAL DE SIMULACION
    # --------------------------------------------------------
    while driver.step() != -1:
        try:
            step_count += 1

            # ====================================================
            # PASO 1: LEER CAMARA Y BUSCAR LINEAS DEL CARRIL
            # ====================================================
            image = get_image(camera)
            if image is None:
                continue  # ignorar frame si la camara esta apagada

            gray, lines = procesar_lineas(image)
            raw_error = calcular_error_direccion(lines, setpoint)

            # Suavizar el error con EMA para evitar volantazos
            if raw_error is not None:
                if smoothed_error is None:
                    smoothed_error = raw_error
                else:
                    smoothed_error = 0.6 * smoothed_error + 0.4 * raw_error
            else:
                smoothed_error = None

            # ====================================================
            # PASO 2: CONTROL PID (mantenerse en el carril)
            # ====================================================
            if smoothed_error is not None:
                # Termino integral (acumula error a lo largo del tiempo)
                integral += smoothed_error
                # Termino derivativo (que tan rapido cambia el error)
                derivative = smoothed_error - prev_error
                # Formula PID: sumamos las tres fuerzas de correccion
                steering = (KP * smoothed_error) + (KI * integral) + (KD * derivative)
                prev_error = smoothed_error
                # Limitar angulo de giro al maximo fisico del vehiculo
                steering = max(-MAX_ANGLE, min(MAX_ANGLE, steering))
            else:
                # Sin deteccion de linea: conducir recto
                steering = 0.0
                integral = 0.0
                prev_error = 0.0

            # ====================================================
            # PASO 3: REVISAR OBSTACULOS CON LIDAR
            # ====================================================
            lidar_angle, lidar_dist = process_lidar(lidar)

            # ====================================================
            # PASO 4: MAQUINA DE ESTADOS
            # ====================================================

            if lidar_dist is not None:
                # Hay obstaculo en rango — clasificar con SVM
                svm_check_cnt += 1
                # Solo pasar el SVM 1 de cada 3 frames para no sobrecargar
                if svm_check_cnt >= 3 or vehicle_state == STATE_NORMAL:
                    svm_check_cnt = 0

                    if svm_model is not None:
                        # Clasificar usando la distancia del LiDAR para adaptar la ventana
                        es_peaton = detect_pedestrian_svm(image, svm_model, lidar_dist)
                    else:
                        # Sin modelo SVM: asumir barril
                        es_peaton = False

                    if es_peaton:
                        vehicle_state = STATE_EMERGENCY_PED
                    else:
                        vehicle_state = STATE_EMERGENCY_OBJ
            else:
                # LiDAR no ve nada — camino libre
                vehicle_state = STATE_NORMAL

            # ====================================================
            # PASO 5: MOVER EL VEHICULO (acelerar / frenar)
            # ====================================================
            aplicar_comandos_vehiculo(driver, vehicle_state, steering, speed, lidar_dist)

            # ====================================================
            # DETECCION Y RECOMENDACIONES EN CONSOLA
            # ====================================================
            if step_count % DEBUG_EVERY == 0:
                dist_str = f"{lidar_dist:.1f}m" if lidar_dist else "---"
                if vehicle_state == STATE_EMERGENCY_PED:
                    print(f"╔══════════════════════════════════════════")
                    print(f"║ [DETECCION] PEATON detectado a {dist_str}")
                    print(f"║  Clasificacion SVM: PERSONA")
                    if lidar_dist and lidar_dist < BRAKE_DIST:
                        print(f"║  Recomendacion: FRENADO DE EMERGENCIA TOTAL")
                        print(f"║  Accion: Vehiculo DETENIDO")
                    else:
                        print(f"║  Recomendacion: REDUCIR VELOCIDAD")
                        print(f"║  Accion: Velocidad reducida a {SLOW_SPEED} km/h")
                    print(f"╚══════════════════════════════════════════")
                elif vehicle_state == STATE_EMERGENCY_OBJ:
                    print(f"╔══════════════════════════════════════════")
                    print(f"║ [DETECCION] OBJETO detectado a {dist_str}")
                    print(f"║  Clasificacion SVM: NO ES PERSONA (barril/caja/cono)")
                    if lidar_dist and lidar_dist < BRAKE_DIST:
                        print(f"║  Recomendacion: FRENADO + LUCES INTERMITENTES")
                        print(f"║  Accion: Vehiculo DETENIDO + hazard flashers ON")
                    else:
                        print(f"║  Recomendacion: REDUCIR VELOCIDAD + INTERMITENTES")
                        print(f"║  Accion: Velocidad {SLOW_SPEED} km/h + hazard flashers ON")
                    print(f"╚══════════════════════════════════════════")
                else:
                    print(f"[NORMAL] Carril libre | Velocidad: {speed} km/h | LiDAR: {dist_str}")

            # ====================================================
            # CONTROLES MANUALES DE VELOCIDAD
            # ====================================================
            key = keyboard.getKey()
            if key == keyboard.UP and speed < MAX_SPEED:
                speed += 5
            elif key == keyboard.DOWN and speed >= 5:
                speed -= 5

        except Exception as e:
            print(f"[ERROR] {e}")
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()
