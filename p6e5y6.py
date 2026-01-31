# modelo_kNN.py
#
# Programa para realizar el entrenamiento de un clasificador k-Nearest Neighbour (kNN) 
# con los motivos de las cartas de póker obtenidos desde un archivo de tipo *.npz.
# Después se realiza la validación del modelo kNN con los motivos de las cartas de test.
#
# Autor: José M Valiente + alumno .............    Fecha: mayo 2023

import cv2
import os
import numpy as np
from clases_cartas import Card, Motif
import warnings

# Definición de etiquetas
FIGURES = ('0','A','2','3','4','5','6','7','8','9','J','Q','K')  # Acceso mediante Card.FIGURES[i]
SUITS = ('Rombos','Picas','Corazones','Treboles')
MOTIF_LABELS = ('Rombos','Picas','Corazones','Treboles','0','2','3','4','5','6','7','8','9','A','J','Q','K','Others')   

# ----------------- TRAINING --------------------
filecard = 'trainCards.npz'
npzfile = np.load(filecard, allow_pickle=True) 
cards = npzfile['Cartas']
llen = cards.size

# Listas vacías para almacenar muestras y etiquetas
samples = []     # Cada muestra es un vector de características (lista de números reales)
responses = []   # Etiqueta real de cada muestra

print("Procesando cartas de entrenamiento...")
j = 0
for i in range(0, llen):   # para todas las cartas
    motifs = cards[i].motifs
    for mot in motifs:
        lbl = mot.motifLabel
        if lbl == 'i':
            continue       # Si el motivo no está etiquetado se descarta
        idx = MOTIF_LABELS.index(lbl)    # Etiqueta real del motivo
        responses.append(idx)
        j += 1
        print(j, idx, lbl)       
        # Añadir a samples el vector de características del motivo.
        # Se asume que en el ejercicio 3 se han almacenado las características en mot.features.
        samples.append(mot.features)

# Convertir las listas en arrays para usarlos en el entrenamiento.
sampl = np.asarray(samples).astype(np.float32)
resp = np.asarray(responses).reshape(-1, 1).astype(np.float32)

# Creación del modelo kNN
knn = cv2.ml.KNearest_create()

# Entrenar el modelo kNN usando ROW_SAMPLE (cada fila es una muestra)
knn.train(sampl, cv2.ml.ROW_SAMPLE, resp)

# ----------------- TEST --------------------
filecardTest = 'testCards.npz'
npzfileT = np.load(filecardTest, allow_pickle=True) 
cardsTest = npzfileT['Cartas']

samplesTest = []
responsesTest = []

print("Procesando cartas de test...")
j = 0
# Se asume que el número de cartas de test es 'le', para evitar conflictos usaremos el tamaño del array
le = cardsTest.size
for i in range(0, le):   # para todas las cartas
    motifs = cardsTest[i].motifs
    for mot in motifs:
        lbl = mot.motifLabel
        # Si el motivo no está etiquetado se asigna 'Others'
        if lbl == 'i':
            lbl = 'Others'
        idx = MOTIF_LABELS.index(lbl)
        responsesTest.append(idx)
        j += 1
        print(j, idx, lbl)
        # Añadir a samplesTest el vector de características del motivo.
        samplesTest.append(mot.features)
     
# Convertir las listas en arrays para usarlos en la predicción.
samplTest = np.asarray(samplesTest).astype(np.float32)
respTest = np.asarray(responsesTest).reshape(-1, 1).astype(np.float32)

# ----------------- PREDICCIÓN con k=3 --------------------
ret, results, neighbours, dist = knn.findNearest(samplTest, 3)

# Visualización de resultados
le_results = len(results)
j = 0
pred = np.zeros(le_results)
real = np.zeros(le_results)

print("\nResultados de la clasificación:")
for i in range(0, le_results):
    pred[i] = int(results[i][0])
    real[i] = respTest[i]
    pred_str = MOTIF_LABELS[int(results[i][0])]
    real_str = MOTIF_LABELS[int(respTest[i])]
    print(f"result: {pred_str}  real:  {real_str}" )
    if pred[i] == real[i]:
        j += 1

print(f'\nTasa de aciertos: {j/le_results:.2f}')

# ----------------- MÉTRICAS DE EVALUACIÓN --------------------
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, ConfusionMatrixDisplay
from sklearn import metrics
import matplotlib.pyplot as plt

CLS_REP = classification_report(real, pred, target_names=MOTIF_LABELS)
print('\nClassification report:\n', CLS_REP) 

CONF_MAT = confusion_matrix(real, pred)
print('Confusion Matrix:\n', CONF_MAT)

MCC = matthews_corrcoef(real, pred)
print('MCC:', MCC)

# Visualización de la matriz de confusión
cm_display = ConfusionMatrixDisplay(confusion_matrix=CONF_MAT, display_labels=MOTIF_LABELS)
cm_display.plot(xticks_rotation='vertical')
plt.show()
