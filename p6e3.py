#!/usr/bin/env python
# segmentat_cartas.py
#
# Programa para realizar operaciones de umbralización global en imágenes en niveles de gris y extracción
# de las cartas de la imagen, obteniendo además los objetos (motivos) y extrayendo sus características.
#
# Autor: José M. Valiente    Fecha: marzo 2023
# Adaptado para el Ejercicio 3 – Extracción del vector de características de cada motivo

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from tkinter import filedialog
import os
from clases_cartas import Card, Motif

def segmentar_objetos_carta(car):
    lowH = 185
    # Umbralizamos la imagen de la carta para detectar motivos
    _, thresh_roi = cv2.threshold(car.grayImage, lowH, 255, cv2.THRESH_BINARY_INV)
    (totalLabels_roi, lab_img, val, centr) = cv2.connectedComponentsWithStats(thresh_roi, 4, cv2.CV_32SC1)
    
    if VISUALIZAR:
        cv2.imshow(window_threshold, thresh_roi)
    
    imot = 0
    MIN_AREA = 1000
    MAX_AREA = 40000
    
    for k in range(1, totalLabels_roi):
        area = val[k, cv2.CC_STAT_AREA]
        if (area > MIN_AREA) and (area < MAX_AREA):
            componentMask2 = (lab_img == k).astype("uint8") * 255
            contours_roi, hi = cv2.findContours(componentMask2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cnt = max(contours_roi, key=lambda k: len(k))   # Contorno de mayor longitud
            
            # Cálculo del bounding box y momentos
            x0, y0, w0, h0 = cv2.boundingRect(cnt)
            mu = cv2.moments(cnt)
            # Calcular centroide (convertido a enteros)
            centroide = np.array([int(mu['m10'] / mu['m00']), int(mu['m01'] / mu['m00'])])
            # Cálculo del círculo mínimo que encierra el contorno
            center, radious = cv2.minEnclosingCircle(cnt)
            center = np.array(list(center), dtype=np.intc)
            radious = int(radious)
            
            # Creación y asignación de datos básicos al motivo
            motif = Motif()
            motif.contour = cnt
            motif.motifId = imot
            motif.area = cv2.contourArea(cnt)
            motif.perimeter = cv2.arcLength(cnt, True)
            motif.centroid = centroide
            motif.moments = mu
            motif.huMoments = cv2.HuMoments(mu).flatten()  # Vector de 7 Hu Moments
            motif.circleCenter = center
            motif.circleRadious = radious
            
            # ------------------ Extracción del vector de características ------------------
            # 1. Relación de aspecto (aspect ratio): ancho/alto del bounding box
            aspect_ratio = float(w0) / h0 if h0 != 0 else 0
            # 2. Extent: relación entre el área del contorno y el área del bounding box
            extent = motif.area / (w0 * h0) if (w0 * h0) != 0 else 0
            # 3. Solidity: relación entre el área del contorno y el área del contorno convexo
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = motif.area / hull_area if hull_area != 0 else 0
            
            # Construir el vector de características (todos números reales o enteros)
            motif.features.append(motif.area)         # Área
            motif.features.append(motif.perimeter)      # Perímetro
            motif.features.append(aspect_ratio)         # Relación de aspecto
            motif.features.append(extent)               # Extent
            motif.features.append(solidity)             # Solidity
            # Agregar los 7 Hu Moments
            for hu in motif.huMoments:
                motif.features.append(float(hu))
            # ---------------------------------------------------------------------------
            
            car.motifs.append(motif)
            imot += 1
            
            # Visualización de resultados si está activada
            if VISUALIZAR:
                img = roi_color.copy()
                cv2.rectangle(img, (x0, y0), (x0+w0, y0+h0), (0, 255, 0), 2)
                cv2.drawContours(img, [cnt], -1, (255, 0, 0), 5)
                cv2.circle(img, (centroide[0], centroide[1]), 4, (255, 0, 0), 2)
                cv2.circle(img, center, radious, (255, 0, 0), 2)
                print(' * Ínidice %d, Área global = %.2f - Nº puntos %d - Área motivo: %.2f - Perímetro: %.2f' %
                      (k, val[k, cv2.CC_STAT_AREA], len(cnt), motif.area, motif.perimeter))
                cv2.imshow(window_roi, img)
                cv2.waitKey()
    print(f'Nº de motivos: {imot} \n')

############### Programa principal  ###############################

VISUALIZAR = False

filecard = 'testCards.npz'   # También puede usarse "trainCards.npz" o "testCards.npz"

window_original = 'Original_image'
window_threshold = 'Thresholded_image'
window_roi = 'ROI image'

cv2.namedWindow(window_original, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
cv2.namedWindow(window_threshold, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
cv2.namedWindow(window_roi, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

low_H = 155

# Lista donde se almacenarán las cartas detectadas
Cards = []

# Selección de una carpeta mediante un diálogo (se espera que contenga imágenes .jpg)
path = filedialog.askdirectory(initialdir="./", title="Seleccione una carpeta")
icard = 0   # Contador para numerar las cartas

for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        if not(name.endswith('.jpg')):
            continue
        filename = os.path.join(root, name)
        img = cv2.imread(filename)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if VISUALIZAR:
            cv2.imshow(window_original, img)
        # Umbralización para segmentar la carta
        ret, thresh1 = cv2.threshold(img_gray, low_H, 255, cv2.THRESH_BINARY_INV)
        (totalLabels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(255 - thresh1, 4, cv2.CV_32S)
       
        # Bucle para cada objeto (carta) detectado en la imagen
        for i in range(1, totalLabels):
            area = values[i, cv2.CC_STAT_AREA]
            if area > 300000:  # Filtro de tamaño para detectar una carta
                componentMask = (label_ids == i).astype("uint8") * 255
                contours, hi = cv2.findContours(componentMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                # Extraer el bounding box de la carta
                x1 = values[i, cv2.CC_STAT_LEFT]
                y1 = values[i, cv2.CC_STAT_TOP]
                w  = values[i, cv2.CC_STAT_WIDTH]
                h  = values[i, cv2.CC_STAT_HEIGHT]
                
                # Recortar la imagen en escala de grises y la imagen original en color de la carta
                roi = img_gray[y1:y1+h, x1:x1+w].copy()
                roi_color = img[y1:y1+h, x1:x1+w].copy()
                rows, cols = roi.shape[:2]
                
                if len(contours) >= 1:
                    c = contours[0]
                    minRect = cv2.minAreaRect(c)
                    ww, hh = minRect[1]
                    if ww < hh:
                        angle = minRect[2]
                    else:
                        angle = minRect[2] - 90
                    
                    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                    roi_warped = cv2.warpAffine(roi, M, (cols, rows))
                    
                    # Crear objeto de tipo Card y asignar sus datos
                    c_obj = Card()
                    c_obj.boundingBox.x = x1
                    c_obj.boundingBox.y = y1
                    c_obj.boundingBox.width = w
                    c_obj.boundingBox.height = h
                    c_obj.angle = angle
                    c_obj.grayImage = roi
                    c_obj.colorImage = roi_color
                    c_obj.cardId = icard
                    
                    # Procesar los motivos de la carta y extraer sus características
                    segmentar_objetos_carta(c_obj)
                    Cards.append(c_obj)
                    icard += 1
        
        if VISUALIZAR:
            print('\n')
            key = -1
            while key == -1:
                key = cv2.pollKey()
                cv2.imshow(window_original, img)
                cv2.imshow(window_threshold, roi)
                cv2.imshow(window_roi, roi_color)
            if key == ord('q') or key == 27:  # 'q' o ESC para salir
                break

# Guardar la lista de cartas en un archivo .npz para uso posterior
np.savez(filecard, Cartas=Cards)
cv2.destroyAllWindows()
