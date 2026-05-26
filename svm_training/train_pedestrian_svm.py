"""
===============================================================================
  Actividad 3.1 — Deteccion de Peatones con SVM (MR4010.10)
  Script de entrenamiento: HOG + SVM (kernel RBF)
===============================================================================

  Descripcion:
      Entrena un clasificador SVM sobre descriptores HOG para distinguir
      peatones de no-peatones (barriles/fondo).
      Adaptado del notebook 3_4_SVM_c del Dr. David Antonio-Torres,
      cambiando la deteccion de vehiculos por deteccion de peatones.

  Dataset:
      - Positivos: imagenes de peatones (carpeta Train/JPEGImages/)
      - Negativos: fondos sin personas (03_Machine_Learning/data_svm/non-vehicles/)
      Si no se dispone de estos datasets, se genera un dataset sintetico
      con skimage.data como fallback.

  Pipeline:
      1. Carga de imagenes → resize a 64x64
      2. Extraccion HOG (11 orientaciones, 16x16 pixels/celda, 2x2 celdas/bloque)
      3. Split 70/30
      4. Entrenamiento SVC (kernel RBF, parametros por defecto)
      5. Evaluacion: classification report + confusion matrix
      6. Exportacion del modelo a model/svm_pedestrian_model.joblib

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
import glob
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACION
# ============================================================

# Tamano de imagen normalizado (todas las imagenes se redimensionan a esto)
IMG_SIZE = (64, 64)

# Parametros HOG (los mismos que usa el profesor en el notebook SVM_c)
HOG_ORIENTATIONS = 11           # bins del histograma de gradientes
HOG_PIXELS_PER_CELL = (16, 16)  # tamano de cada celda en pixeles
HOG_CELLS_PER_BLOCK = (2, 2)    # celdas por bloque para normalizacion

# Rutas del dataset
# Positivos: imagenes de peatones (JPEG)
POS_DIR = os.path.join(os.path.dirname(__file__), "Train", "JPEGImages")
# Negativos: fondos sin personas (PNG del dataset del profesor)
NEG_PATTERN = os.path.join(
    os.path.dirname(__file__),
    "03_Machine_Learning", "data_svm", "non-vehicles", "**", "*.png"
)

# Ruta del modelo exportado
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "svm_pedestrian_model.joblib")

# Ruta de la confusion matrix
SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "screenshots")

# Semilla para reproducibilidad
RANDOM_STATE = 42


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def extract_hog_features(img_path):
    """
    Carga una imagen, la normaliza a 64x64 y extrae su descriptor HOG.
    Retorna el vector de features o None si la imagen no se puede leer.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        transform_sqrt=False,
        visualize=False,
        feature_vector=True,
    )
    return features


def load_positive_paths():
    """
    Busca todas las imagenes de peatones en Train/JPEGImages/.
    Acepta cualquier extension que OpenCV pueda leer.
    """
    paths = []
    if not os.path.isdir(POS_DIR):
        return paths
    for root, dirs, files in os.walk(POS_DIR):
        for f in files:
            full = os.path.join(root, f)
            # solo agregar si OpenCV puede abrirla (descarta archivos no-imagen)
            if cv2.imread(full) is not None:
                paths.append(full)
    return paths


def load_negative_paths():
    """
    Busca todas las imagenes negativas (fondos sin personas).
    Usa el patron glob definido en NEG_PATTERN.
    """
    return glob.glob(NEG_PATTERN, recursive=True)


def generate_synthetic_dataset():
    """
    Fallback: genera un dataset sintetico si los datasets reales
    no estan disponibles. Usa imagenes de skimage.data.
    """
    from skimage import data
    rng = np.random.default_rng(RANDOM_STATE)

    print("[WARN] Datasets no encontrados, generando dataset sintetico...")
    print(f"[WARN] Para mejores resultados, coloca los datos en:")
    print(f"       Positivos: {POS_DIR}")
    print(f"       Negativos: 03_Machine_Learning/data_svm/non-vehicles/")

    # Positivos: usar astronaut como base y crear variaciones
    astronaut = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2GRAY)
    X_pos = []
    for _ in range(200):
        h, w = astronaut.shape
        y = rng.integers(0, max(1, h - 64))
        x = rng.integers(0, max(1, w - 64))
        crop = astronaut[y:y + 64, x:x + 64]
        noise = rng.integers(-15, 15, size=crop.shape, dtype=np.int16)
        noisy = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        features = hog(
            cv2.resize(noisy, IMG_SIZE),
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            transform_sqrt=False,
            feature_vector=True,
        )
        X_pos.append(features)

    # Negativos: texturas sin personas
    textures = [data.brick(), data.grass(), data.gravel()]
    X_neg = []
    for tex in textures:
        if len(tex.shape) == 3:
            tex = cv2.cvtColor(tex, cv2.COLOR_RGB2GRAY)
        for _ in range(70):
            h, w = tex.shape
            y = rng.integers(0, max(1, h - 64))
            x = rng.integers(0, max(1, w - 64))
            crop = tex[y:y + 64, x:x + 64]
            features = hog(
                cv2.resize(crop, IMG_SIZE),
                orientations=HOG_ORIENTATIONS,
                pixels_per_cell=HOG_PIXELS_PER_CELL,
                cells_per_block=HOG_CELLS_PER_BLOCK,
                transform_sqrt=False,
                feature_vector=True,
            )
            X_neg.append(features)

    print(f"[INFO] Positivos sinteticos: {len(X_pos)}")
    print(f"[INFO] Negativos sinteticos: {len(X_neg)}")
    return np.array(X_pos), np.array(X_neg)


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def main():
    print("=" * 55)
    print("  ENTRENAMIENTO SVM — DETECCION DE PEATONES")
    print("=" * 55)

    # --- Paso 1: Cargar dataset ---
    pos_paths = load_positive_paths()
    neg_paths = load_negative_paths()

    print(f"\nImagenes de peatones encontradas: {len(pos_paths)}")
    print(f"Imagenes sin peatones encontradas: {len(neg_paths)}")

    use_synthetic = len(pos_paths) == 0 or len(neg_paths) == 0

    if not use_synthetic:
        # --- Paso 2: Extraer HOG de imagenes reales ---
        print("\nExtrayendo HOG de peatones...")
        pos_feats = [f for p in pos_paths if (f := extract_hog_features(p)) is not None]
        X_pos = np.vstack(pos_feats).astype(np.float64)
        print(f"  Shape: {X_pos.shape}")

        print("Extrayendo HOG de no-peatones...")
        neg_feats = [f for p in neg_paths if (f := extract_hog_features(p)) is not None]
        X_neg = np.vstack(neg_feats).astype(np.float64)
        print(f"  Shape: {X_neg.shape}")
    else:
        X_pos, X_neg = generate_synthetic_dataset()

    # Etiquetas: 1 = peaton, 0 = no-peaton
    y_pos = np.ones(len(X_pos))
    y_neg = np.zeros(len(X_neg))

    # Combinar datasets
    X = np.vstack((X_pos, X_neg))
    y = np.hstack((y_pos, y_neg))
    print(f"\nDataset total: {X.shape}")

    # --- Paso 3: Split 70/30 ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )
    print(f"Entrenamiento: {X_train.shape}  |  Prueba: {X_test.shape}")

    # --- Paso 4: Entrenar SVM (kernel RBF por defecto) ---
    print("\nEntrenando SVM (puede tardar unos minutos)...")
    svc_model = SVC()
    svc_model.fit(X_train, y_train)
    print("Entrenamiento terminado.")

    # --- Paso 5: Evaluacion ---
    y_pred = svc_model.predict(X_test)

    # Confusion matrix como heatmap
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)

    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d",
                    xticklabels=["No peaton", "Peaton"],
                    yticklabels=["No peaton", "Peaton"], ax=ax)
        ax.set_title("Matriz de confusion — SVM Deteccion de Peatones")
        ax.set_ylabel("Real")
        ax.set_xlabel("Predicho")
    except ImportError:
        # fallback sin seaborn
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.matshow(cm, cmap="Blues")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, str(val), ha="center", va="center", fontsize=14)
        ax.set_xticklabels(["", "No peaton", "Peaton"])
        ax.set_yticklabels(["", "No peaton", "Peaton"])
        ax.set_title("Matriz de confusion — SVM Deteccion de Peatones")
        ax.set_ylabel("Real")
        ax.set_xlabel("Predicho")

    fig.tight_layout()
    cm_path = os.path.join(SCREENSHOTS_DIR, "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"\nImagen guardada: {cm_path}")

    # Reporte de clasificacion
    print("\n--- Reporte de clasificacion ---")
    print(classification_report(y_test, y_pred,
                                target_names=["No peaton", "Peaton"]))

    # --- Paso 6: Exportar modelo ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(svc_model, MODEL_PATH)
    print(f"Modelo exportado: {MODEL_PATH}")
    print("Copia este archivo a la carpeta del controlador si es necesario.")


if __name__ == "__main__":
    main()
