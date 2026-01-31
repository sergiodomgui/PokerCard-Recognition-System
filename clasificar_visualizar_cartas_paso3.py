import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from clases_cartas import Card, Motif
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import os
import math

# --- Selector de Clasificador ---
# Elige entre: 'KNN', 'SVM', 'RTREES' (Random Forest), 'NN' (Red Neuronal)
CLASSIFIER_TYPE = 'NN' 

# --- Importaciones condicionales para la Red Neuronal ---
if CLASSIFIER_TYPE == 'NN':
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        print("ERROR: Para usar la Red Neuronal (NN), necesitas instalar TensorFlow.")
        print("Ejecuta: pip install tensorflow")
        exit()

# --- CONSTANTES Y CONFIGURACIÓN ---
MOTIF_LABELS = ('Rombos', 'Picas', 'Corazones', 'Treboles', '0', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'J', 'Q', 'K', 'Others')
SUITS = ('Rombos', 'Picas', 'Corazones', 'Treboles')
FIGURES_MOTIFS = ('0', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'J', 'Q', 'K')
MOTIF_TO_FIGURE_MAP = {
    '0': '0', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
    'A': 'A', 'J': 'J', 'Q': 'Q', 'K': 'K'
}

FILE_TRAIN = 'trainCards.npz'
FILE_TEST = 'testCards.npz'

# --- 1. CARGA DE DATOS DE ENTRENAMIENTO ---
print("--- Paso 1: Cargando datos de entrenamiento ---")
if not os.path.exists(FILE_TRAIN):
    print(f"Error: El archivo de entrenamiento '{FILE_TRAIN}' no se encontró.")
    exit()
npzfile_train = np.load(FILE_TRAIN, allow_pickle=True)
cards_train = npzfile_train['Cartas']
samples_train, responses_train = [], []
for card in cards_train:
    for motif in card.motifs:
        if motif.huMoments is not None and len(motif.huMoments) == 7 and motif.motifLabel != 'i':
            samples_train.append(motif.huMoments)
            try:
                responses_train.append(MOTIF_LABELS.index(motif.motifLabel))
            except ValueError:
                responses_train.append(MOTIF_LABELS.index('Others'))

samples_train = np.array(samples_train, dtype=np.float32)
responses_train = np.array(responses_train)

# Escalar datos de entrenamiento
scaler = StandardScaler()
samples_train_scaled = scaler.fit_transform(samples_train)

# --- 2. ENTRENAMIENTO DEL CLASIFICADOR ---
print(f"\n--- Paso 2: Entrenando el clasificador de motivos: {CLASSIFIER_TYPE} ---")
model = None

if CLASSIFIER_TYPE == 'KNN':
    model = cv2.ml.KNearest_create()
    model.setDefaultK(3)
    model.setIsClassifier(True)
    model.train(samples_train_scaled, cv2.ml.ROW_SAMPLE, responses_train.astype(np.float32).reshape(-1, 1))

elif CLASSIFIER_TYPE == 'SVM':
    model = cv2.ml.SVM_create()
    model.setType(cv2.ml.SVM_C_SVC)
    model.setKernel(cv2.ml.SVM_RBF) # Kernel Radial Basis Function, bueno para datos no lineales
    model.setC(10) # Parámetro de regularización
    model.setGamma(0.1) # Parámetro del kernel
    model.train(samples_train_scaled, cv2.ml.ROW_SAMPLE, responses_train.astype(np.int32))

elif CLASSIFIER_TYPE == 'RTREES':
    model = cv2.ml.RTrees_create()
    model.setMaxDepth(20) # Profundidad máxima de los árboles
    model.setMinSampleCount(5) # Mínimo de muestras para dividir un nodo
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 0.01))
    model.train(samples_train_scaled, cv2.ml.ROW_SAMPLE, responses_train.astype(np.int32))

elif CLASSIFIER_TYPE == 'NN':
    input_dim = samples_train_scaled.shape[1]
    num_classes = len(MOTIF_LABELS)
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    history = model.fit(samples_train_scaled, responses_train,
                        epochs=100,
                        batch_size=32,
                        validation_split=0.2,
                        verbose=1)
else:
    print(f"Error: Tipo de clasificador '{CLASSIFIER_TYPE}' no reconocido.")
    exit()

print("Entrenamiento completado.")

# --- 3. PREDICCIÓN Y EVALUACIÓN DE MOTIVOS (MATRIZ ORIGINAL) ---
print(f"\n--- Paso 3: Evaluando el clasificador de MOTIVOS ({CLASSIFIER_TYPE}) ---")
if not os.path.exists(FILE_TEST):
    print(f"Error: El archivo de prueba '{FILE_TEST}' no se encontró.")
    exit()
npzfile_test = np.load(FILE_TEST, allow_pickle=True)
cards_test = npzfile_test['Cartas']

all_samples_test, all_responses_test = [], []
for card in cards_test:
    for motif in card.motifs:
        if motif.huMoments is not None and len(motif.huMoments) == 7:
            all_samples_test.append(motif.huMoments)
            try:
                label = motif.motifLabel if motif.motifLabel != 'i' else 'Others'
                all_responses_test.append(MOTIF_LABELS.index(label))
            except ValueError:
                all_responses_test.append(MOTIF_LABELS.index('Others'))

if all_samples_test:
    all_samples_test_scaled = scaler.transform(np.array(all_samples_test, dtype=np.float32))
    all_real_motifs_idx = np.array(all_responses_test)
    
    # Predecir todos los motivos a la vez
    if CLASSIFIER_TYPE == 'KNN':
        _, results, _, _ = model.findNearest(all_samples_test_scaled, 3)
        all_predicted_motifs_idx = results.flatten()
    elif CLASSIFIER_TYPE == 'SVM' or CLASSIFIER_TYPE == 'RTREES':
        _, results = model.predict(all_samples_test_scaled)
        all_predicted_motifs_idx = results.flatten()
    elif CLASSIFIER_TYPE == 'NN':
        probabilities = model.predict(all_samples_test_scaled)
        all_predicted_motifs_idx = np.argmax(probabilities, axis=1)

    print("\n--- Informe de Clasificación para Motivos Individuales ---")
    print(classification_report(all_real_motifs_idx, all_predicted_motifs_idx, target_names=MOTIF_LABELS, zero_division=0))
    
    conf_mat_motifs = confusion_matrix(all_real_motifs_idx, all_predicted_motifs_idx, labels=range(len(MOTIF_LABELS)))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_mat_motifs, display_labels=MOTIF_LABELS)
    
    fig_cm, ax_cm = plt.subplots(figsize=(10, 10))
    cm_display.plot(cmap=plt.cm.Blues, ax=ax_cm, xticks_rotation='vertical')
    plt.title(f'Matriz de Confusión de Motivos ({CLASSIFIER_TYPE})')
    fig_cm.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.show()

# --- 4. CLASIFICACIÓN CARTA A CARTA PARA VISUALIZACIÓN ---
print("\n--- Paso 4: Clasificación carta a carta y cálculo de acierto general ---")

correctly_classified_cards = 0
total_cards = len(cards_test)

for card in cards_test:
    predicted_suits, predicted_figures = [], []
    valid_motifs_in_card = [m for m in card.motifs if m.huMoments is not None and len(m.huMoments) == 7]

    if not valid_motifs_in_card:
        card.predictedSuit = 'No Motivos'
        card.predictedFigure = 'No Motivos'
    else:
        motifs_to_predict = np.array([m.huMoments for m in valid_motifs_in_card], dtype=np.float32)
        motifs_to_predict_scaled = scaler.transform(motifs_to_predict)
        
        # Predicción para los motivos de esta carta
        if CLASSIFIER_TYPE == 'KNN':
            _, results, _, _ = model.findNearest(motifs_to_predict_scaled, 3)
            predicted_labels_idx_motifs = results.flatten()
        elif CLASSIFIER_TYPE == 'SVM' or CLASSIFIER_TYPE == 'RTREES':
            _, results = model.predict(motifs_to_predict_scaled)
            predicted_labels_idx_motifs = results.flatten()
        elif CLASSIFIER_TYPE == 'NN':
            probabilities = model.predict(motifs_to_predict_scaled, verbose=0)
            predicted_labels_idx_motifs = np.argmax(probabilities, axis=1)

        for pred_idx in predicted_labels_idx_motifs:
            predicted_motif_label = MOTIF_LABELS[int(pred_idx)]
            if predicted_motif_label in SUITS:
                predicted_suits.append(predicted_motif_label)
            elif predicted_motif_label in FIGURES_MOTIFS:
                predicted_figures.append(MOTIF_TO_FIGURE_MAP[predicted_motif_label])

        card.predictedSuit = Counter(predicted_suits).most_common(1)[0][0] if predicted_suits else 'Desconocido'
        card.predictedFigure = Counter(predicted_figures).most_common(1)[0][0] if predicted_figures else 'Desconocido'
    
    is_correct = (card.predictedSuit == card.realSuit) and (card.predictedFigure == card.realFigure)
    
    if is_correct:
        correctly_classified_cards += 1
        
    print(f"  Carta ID: {card.cardId} -> Real: {card.realFigure} de {card.realSuit} | Predicción: {card.predictedFigure} de {card.predictedSuit} {'-> CORRECTO' if is_correct else '-> INCORRECTO'}")

# --- 5. RESUMEN FINAL Y VISUALIZACIÓN GRÁFICA ---
print("\n--- Paso 5: Resumen Final y Visualización Gráfica ---")

if total_cards > 0:
    accuracy = (correctly_classified_cards / total_cards) * 100
    print("-" * 50)
    print("                    RESUMEN GENERAL")
    print("-" * 50)
    print(f"Clasificador Utilizado: {CLASSIFIER_TYPE}")
    print(f"Cartas totales analizadas: {total_cards}")
    print(f"Cartas clasificadas correctamente: {correctly_classified_cards}")
    print(f"TASA DE ACIERTO GENERAL: {accuracy:.2f}%")
    print("-" * 50)
else:
    print("No se procesaron cartas para calcular la precisión.")

if total_cards > 0:
    cols = 5 
    rows = math.ceil(total_cards / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()

    for i, card in enumerate(cards_test):
        img = card.colorImage.copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        real_label = f"Real: {card.realFigure} de {card.realSuit}"
        pred_label = f"Pred: {card.predictedFigure} de {card.predictedSuit}"
        is_correct = (card.predictedSuit == card.realSuit) and (card.predictedFigure == card.realFigure)
        
        border_color = (0, 255, 0) if is_correct else (255, 0, 0)
        img_rgb = cv2.copyMakeBorder(img_rgb, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=border_color)

        axes[i].imshow(img_rgb)
        axes[i].set_title(f"{real_label}\n{pred_label}", color= 'green' if is_correct else 'red')
        axes[i].axis('off')

    for j in range(total_cards, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.suptitle('Resultados de la Clasificación de Cartas', fontsize=16, y=1.02)
    plt.show()
