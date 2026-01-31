import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from tkinter import filedialog
import os
from clases_cartas import Card, Motif

def segmentar_objetos_carta(car, roi_color):
    """
    Segmenta los motivos (palos y figuras) de una única imagen de carta.

    Esta función binariza la imagen de la carta, encuentra los contornos de los motivos,
    y calcula un vector de características para cada uno, centrado en los Momentos de Hu
    transformados logarítmicamente.
    """
    # --- AJUSTE CLAVE: UMBRAL DE BINARIZACIÓN ---
    lowH = 185
    _, thresh_roi = cv2.threshold(car.grayImage, lowH, 255, cv2.THRESH_BINARY_INV)

    if VISUALIZAR:
        cv2.imshow(window_threshold, thresh_roi)

    (totalLabels_roi, lab_img, val, centr) = cv2.connectedComponentsWithStats(thresh_roi, 4, cv2.CV_32SC1)
    
    imot = 0
    # --- AJUSTE CLAVE: FILTROS DE ÁREA PARA MOTIVOS ---
    MIN_AREA = 1000
    MAX_AREA = 40000
    
    for k in range(1, totalLabels_roi):
        area = val[k, cv2.CC_STAT_AREA]
        if (area > MIN_AREA) and (area < MAX_AREA):
            componentMask = (lab_img == k).astype("uint8") * 255
            contours_roi, _ = cv2.findContours(componentMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours_roi:
                continue

            cnt = max(contours_roi, key=cv2.contourArea)
            
            mu = cv2.moments(cnt)
            if mu['m00'] == 0: # Evitar división por cero si el área del momento es 0
                continue

            # --- CÁLCULO Y TRANSFORMACIÓN DE MOMENTOS DE HU (CRÍTICO) ---
            huMoments_raw = cv2.HuMoments(mu).flatten()
            # Aplicamos la transformación logarítmica para escalar los valores y hacerlos útiles.
            # Se añade un pequeño épsilon (1e-7) para evitar errores de log(0).
            huMoments_transformed = -np.sign(huMoments_raw) * np.log10(np.abs(huMoments_raw) + 1e-7)
            
            # Creación del objeto Motivo
            motif = Motif()
            motif.motifId = imot
            motif.contour = cnt
            motif.area = cv2.contourArea(cnt)
            motif.perimeter = cv2.arcLength(cnt, True)
            
            # Guardamos tanto los momentos transformados como el vector de características
            # Usaremos consistentemente los momentos de Hu transformados en todo el sistema.
            motif.huMoments = huMoments_transformed
            motif.features = huMoments_transformed

            car.motifs.append(motif)
            imot += 1
            
            if VISUALIZAR:
                x0, y0, w0, h0 = cv2.boundingRect(cnt)
                img_vis = roi_color.copy()
                cv2.drawContours(img_vis, [cnt], -1, (255, 0, 0), 3)
                cv2.rectangle(img_vis, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 2)
                cv2.imshow(window_roi, img_vis)
                cv2.waitKey(0)

    print(f'Nº de motivos detectados en la carta: {imot}')

############### Programa principal ###############################

VISUALIZAR = False  # Cambia a True para depurar visualmente la segmentación
filecard = 'testCards.npz' # Asegúrate de que sea 'trainCards.npz' o 'testCards.npz' según corresponda

window_original = 'Original_image'
window_threshold = 'Thresholded_image'
window_roi = 'ROI image'
if VISUALIZAR:
    cv2.namedWindow(window_original, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_threshold, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_roi, cv2.WINDOW_NORMAL)

low_H = 155
Cards = []

path = filedialog.askdirectory(initialdir=".", title="Seleccione la carpeta con las imágenes de cartas")
if not path:
    print("No se seleccionó ninguna carpeta. Saliendo.")
    exit()

icard = 0
for root, _, files in os.walk(path):
    for name in files:
        if not name.lower().endswith('.jpg'):
            continue
        
        filename = os.path.join(root, name)
        img = cv2.imread(filename)
        if img is None:
            print(f"Advertencia: No se pudo cargar la imagen {filename}.")
            continue
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if VISUALIZAR:
            cv2.imshow(window_original, img)

        _, thresh1 = cv2.threshold(img_gray, low_H, 255, cv2.THRESH_BINARY_INV)
        (totalLabels, label_ids, values, _) = cv2.connectedComponentsWithStats(255 - thresh1, 4, cv2.CV_32S)

        for i in range(1, totalLabels):
            area = values[i, cv2.CC_STAT_AREA]
            if area > 300000:  # Filtro de tamaño para detectar una carta entera
                x1, y1, w, h = values[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
                
                roi_gray = img_gray[y1:y1+h, x1:x1+w].copy()
                roi_color = img[y1:y1+h, x1:x1+w].copy()
                
                c_obj = Card()
                c_obj.grayImage = roi_gray
                c_obj.colorImage = roi_color
                c_obj.cardId = icard
                
                segmentar_objetos_carta(c_obj, roi_color)
                Cards.append(c_obj)
                icard += 1
        
        if VISUALIZAR:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

np.savez(filecard, Cartas=np.array(Cards, dtype=object))
print(f"\nProceso finalizado. Se han guardado {len(Cards)} cartas en el archivo '{filecard}'.")
cv2.destroyAllWindows()
