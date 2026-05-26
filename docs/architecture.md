# Arquitectura del Sistema — Actividad 3.1

## Diagrama de bloques

```
┌─────────────────────────────────────────────────────────────┐
│                    CONTROLADOR AUTONOMO                      │
│                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │  Camara   │───>│ Lane Follow  │───>│   Controlador     │  │
│  │ 256x128   │    │ Canny+Hough  │    │      PID          │  │
│  └──────────┘    └──────────────┘    │ Kp=0.008 Kd=0.015 │  │
│       │                              └────────┬──────────┘  │
│       │                                       │              │
│       │          ┌──────────────┐              v              │
│       │          │   LiDAR      │      ┌─────────────┐      │
│       │          │ Sick LMS 291 │─────>│  Vehiculo    │      │
│       │          │  (180° FOV)  │      │  steering +  │      │
│       │          └──────┬───────┘      │  speed       │      │
│       │                 │              └─────────────┘      │
│       │                 v                                    │
│       │          ┌──────────────┐                            │
│       │          │  Obstaculo   │                            │
│       │          │  < 20m en    │                            │
│       │          │  ±20° centro │                            │
│       │          └──────┬───────┘                            │
│       │                 │ SI                                  │
│       v                 v                                    │
│  ┌──────────────────────────┐                                │
│  │   Sliding Window + HOG   │                                │
│  │   → SVM (SVC, RBF)      │                                │
│  └────────────┬─────────────┘                                │
│               │                                              │
│        ┌──────┴──────┐                                       │
│        v             v                                       │
│  ┌──────────┐  ┌──────────┐                                  │
│  │  PEATON  │  │  OBJETO  │                                  │
│  │ Reducir  │  │ Reducir  │                                  │
│  │ vel/Fren │  │ vel/Fren │                                  │
│  │          │  │ +Flashers│                                  │
│  └──────────┘  └──────────┘                                  │
└─────────────────────────────────────────────────────────────┘
```

## Pipeline de vision (Lane Following)

```
Camara (256x128 BGRA)
    │
    v
Escala de grises (cv2.COLOR_BGR2GRAY)
    │
    v
Gaussian Blur (5x5)
    │
    v
Canny Edge Detection (50, 150)
    │
    v
ROI Trapezoidal (mitad inferior)
    │
    v
Hough Lines P (rho=1, theta=π/180, threshold=15)
    │
    v
Calculo de error (midpoint vs setpoint)
    │
    v
EMA Smoothing (alpha=0.6)
    │
    v
PID (Kp=0.008, Ki=0, Kd=0.015)
    │
    v
Steering angle (saturado a ±0.5 rad)
```

## Pipeline de deteccion (LiDAR + SVM)

```
LiDAR Sick LMS 291 (180° FOV, ~180 puntos)
    │
    v
Filtrar rango central (±20 indices ≈ ±20°)
    │
    v
¿avg_distance < 20m?
    │
    ├── NO → continuar lane following a 30 km/h
    │
    └── SI → activar clasificacion SVM
              │
              v
         Camara → frame BGRA
              │
              v
         ROI: 25% a 85% de altura
              │
              v
         Sliding window adaptativa:
           >14m → 32x32, stride 20
           8-14m → 48x48, stride 24
           <8m  → 64x64, stride 32
              │
              v
         Resize a 64x64 → grises
              │
              v
         HOG (11 orientaciones, 16x16 px/celda, 2x2 celdas/bloque)
              │
              v
         SVC.decision_function() > 0.55 → PEATON
              │
              v
         ≥1 ventana "peaton" → STATE_EMERGENCY_PED
         0 ventanas "peaton" → STATE_EMERGENCY_OBJ
```

## Parametros del PID

| Parametro | Valor | Justificacion |
|-----------|-------|---------------|
| Kp | 0.008 | Correccion proporcional suave para evitar oscilaciones |
| Ki | 0.0 | Desactivado — causaba oscilaciones fuertes en pruebas |
| Kd | 0.015 | Amortiguamiento ante cambios bruscos de error |
| EMA alpha | 0.6 | Balance entre reactividad y suavidad |
| Max angle | ±0.5 rad | Limite fisico del volante del vehiculo |

## Flujo de decision ante obstaculos

```
                    ┌─────────────┐
                    │ LiDAR scan  │
                    └──────┬──────┘
                           │
                    ┌──────v──────┐
                    │ dist < 20m? │
                    └──────┬──────┘
                     NO    │    SI
                ┌──────────┤
                v          v
          Seguir PID   Clasificar SVM
          30 km/h          │
                    ┌──────┴──────┐
                    │  ¿Peaton?   │
                    └──────┬──────┘
                     NO    │    SI
                ┌──────────┤
                v          v
           OBJETO       PEATON
           +Flashers    (sin flashers)
                │          │
                v          v
           ┌───────────────────┐
           │  ¿dist < 8m?     │
           └────────┬──────────┘
              NO    │    SI
           ┌────────┤
           v        v
        10 km/h   Frenado total
        (reducir)  (0 km/h)
```

## Salida en consola

El controlador imprime en tiempo real la informacion de cada deteccion:

```
╔══════════════════════════════════════════
║ [DETECCION] PEATON detectado a 14.6m
║  Clasificacion SVM: PERSONA
║  Recomendacion: REDUCIR VELOCIDAD
║  Accion: Velocidad reducida a 10 km/h
╚══════════════════════════════════════════

╔══════════════════════════════════════════
║ [DETECCION] OBJETO detectado a 6.7m
║  Clasificacion SVM: NO ES PERSONA (barril/caja/cono)
║  Recomendacion: FRENADO + LUCES INTERMITENTES
║  Accion: Vehiculo DETENIDO + hazard flashers ON
╚══════════════════════════════════════════
```

## Dataset de entrenamiento

Imagenes del dataset proporcionado en los materiales del modulo:
- **Positivos:** imagenes de peatones (`Train/JPEGImages/`) redimensionadas a 64x64
- **Negativos:** fondos sin personas (`data_svm/non-vehicles/`) redimensionadas a 64x64
- **Fallback:** dataset sintetico generado con `skimage.data` si los datos no estan disponibles

## Tecnologias

- **Simulador:** Webots R2023b+
- **Lenguaje:** Python 3.10+
- **Vision:** OpenCV 4.x
- **ML:** scikit-learn (SVC, kernel RBF), scikit-image (HOG)
- **Sensores:** Camara (256x128), Sick LMS 291 LiDAR (180° FOV)
