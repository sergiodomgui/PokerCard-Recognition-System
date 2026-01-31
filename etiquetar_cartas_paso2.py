import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import ImageTk, Image
import cv2
import os
import numpy as np
from clases_cartas import Card, Motif
import warnings

FIGURES = ('0','A','2','3','4','5','6','7','8','9','J','Q','K') # Se accede mediantge Carta.FIGURES[i]
SUITS = ('Rombos','Picas','Corazones','Treboles')
MOTIF_LABELS = ('Rombos','Picas','Corazones','Treboles','0','2','3','4','5','6','7','8','9','A','J','Q','K','Others')

# --- CONFIGURACIÓN ---
filecard = 'testCards.npz' # trainCards.npz y testCards.npz. ¡Cambia esto según el conjunto que quieras etiquetar!

# Umbral de similitud para la función "Aplicar a similares en esta carta"
HU_MOMENT_SIMILARITY_THRESHOLD = 0.8 # Ajusta este valor según la variación de tus motivos

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Herramienta de Etiquetado de Cartas')
        self.resizable(True, True)
        
        self.cards = []
        self.card_idx = 0
        self.motif_idx = 0
        self.original_motif_labels = {} # Para almacenar etiquetas originales para comparación

        self.load_data()
        if not self.cards:
            messagebox.showerror("Error", f"No se encontraron cartas en {filecard}. Asegúrate de que el archivo existe y contiene datos.")
            self.destroy()
            return
        
        self.create_widgets()
        self.update_display()
        
        # --- Atajos de teclado ---
        self.bind_all('<Key>', self.on_key_press)
        print("\n--- ATENCION: ATAJOS DE TECLADO ---")
        print("  's' / 'S': Guardar carta actual")
        print("  'n': Siguiente motivo")
        print("  'p': Motivo anterior")
        print("  'N': Siguiente carta")
        print("  'P': Carta anterior")
        print("  'q' / 'Q' / 'Escape': Salir")
        print("----------------------------------\n")


    def load_data(self):
        try:
            npzfile = np.load(filecard, allow_pickle=True)
            self.cards = npzfile['Cartas'].tolist() # Convertir a lista si es un array de numpy
            # Almacenar las etiquetas originales al cargar para detectar cambios
            for c_idx, card in enumerate(self.cards):
                for m_idx, motif in enumerate(card.motifs):
                    self.original_motif_labels[(c_idx, m_idx)] = motif.motifLabel
            print(f"Cargadas {len(self.cards)} cartas desde {filecard}")
        except FileNotFoundError:
            print(f"ERROR: Archivo {filecard} no encontrado. Por favor, ejecuta 'segmentar_cartas.py' primero.")
            self.cards = []
        except Exception as e:
            print(f"ERROR al cargar {filecard}: {e}")
            self.cards = []

    def create_widgets(self):
        # Frame principal
        main_frame = ttk.Frame(self, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # --- Etiquetado de Motivos ---
        lf_motif = ttk.LabelFrame(main_frame, text='Etiquetado de Motivos')
        lf_motif.grid(column=0, row=0, padx=5, pady=5, sticky=(tk.W, tk.E))

        self.motif_label = ttk.Label(lf_motif, text="Motivo:")
        self.motif_label.grid(column=0, row=0, padx=5, pady=5)

        self.motif_combo = ttk.Combobox(lf_motif, values=MOTIF_LABELS, state="readonly")
        self.motif_combo.grid(column=1, row=0, padx=5, pady=5)
        self.motif_combo.bind("<<ComboboxSelected>>", self.on_motif_label_selected)

        self.prevMotif_button = ttk.Button(lf_motif, text='Motivo anterior (p)', command=self.prev_motif)
        self.prevMotif_button.grid(column=0, row=1, padx=5, pady=5)
        self.nextMotif_button = ttk.Button(lf_motif, text='Siguiente motivo (n)', command=self.next_motif)
        self.nextMotif_button.grid(column=1, row=1, padx=5, pady=5)

        # Botón "Aplicar a similares en esta carta"
        self.apply_similar_button = ttk.Button(lf_motif, text='Aplicar a similares en esta carta', command=self.apply_to_similar_motifs)
        self.apply_similar_button.grid(column=0, row=2, columnspan=2, padx=5, pady=5)

        # --- Etiquetado de Carta ---
        lf_card = ttk.LabelFrame(main_frame, text='Etiquetado de Carta')
        lf_card.grid(column=0, row=1, padx=5, pady=5, sticky=(tk.W, tk.E))

        self.card_suit_label = ttk.Label(lf_card, text="Palo Real:")
        self.card_suit_label.grid(column=0, row=0, padx=5, pady=5)
        self.card_suit_combo = ttk.Combobox(lf_card, values=SUITS, state="readonly")
        self.card_suit_combo.grid(column=1, row=0, padx=5, pady=5)
        self.card_suit_combo.bind("<<ComboboxSelected>>", self.on_card_suit_selected)

        self.card_figure_label = ttk.Label(lf_card, text="Figura Real:")
        self.card_figure_label.grid(column=0, row=1, padx=5, pady=5)
        self.card_figure_combo = ttk.Combobox(lf_card, values=FIGURES, state="readonly")
        self.card_figure_combo.grid(column=1, row=1, padx=5, pady=5)
        self.card_figure_combo.bind("<<ComboboxSelected>>", self.on_card_figure_selected)

        # --- Navegación y Guardar ---
        lf_nav_save = ttk.LabelFrame(main_frame, text='Navegación y Guardar')
        lf_nav_save.grid(column=0, row=2, padx=5, pady=5, sticky=(tk.W, tk.E))

        self.save_button = ttk.Button(lf_nav_save, text='Guardar carta (s)', command=self.save_current_card)
        self.save_button.grid(column=0, row=0, padx=5, pady=5)

        self.prevCard_button = ttk.Button(lf_nav_save, text='Carta anterior (P)', command=self.prev_card)
        self.prevCard_button.grid(column=1, row=0, padx=5, pady=5)
        self.nextCard_button = ttk.Button(lf_nav_save, text='Siguiente carta (N)', command=self.next_card)
        self.nextCard_button.grid(column=2, row=0, padx=5, pady=5)

        self.exit_button = ttk.Button(lf_nav_save, text='Salir (q/Esc)', command=self.exit_app)
        self.exit_button.grid(column=0, row=1, columnspan=3, padx=5, pady=5)

        # --- Visualización de Imágenes ---
        self.card_image_label = ttk.Label(main_frame)
        self.card_image_label.grid(column=1, row=0, rowspan=3, padx=10, pady=5, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        self.motif_image_label = ttk.Label(main_frame)
        self.motif_image_label.grid(column=2, row=0, rowspan=3, padx=10, pady=5, sticky=(tk.N, tk.S, tk.E, tk.W))

    def on_key_press(self, event):
        key = event.keysym
        if key == 'n':
            self.next_motif()
        elif key == 'p':
            self.prev_motif()
        elif key == 'N':
            self.next_card()
        elif key == 'P':
            self.prev_card()
        elif key == 's' or key == 'S':
            self.save_current_card()
        elif key == 'q' or key == 'Q' or key == 'Escape':
            self.exit_app()

    def update_display(self):
        if not self.cards:
            self.card_image_label.config(image=None)
            self.motif_image_label.config(image=None)
            self.motif_combo.set('')
            self.card_suit_combo.set('')
            self.card_figure_combo.set('')
            self.title('Herramienta de Etiquetado de Cartas - Sin datos')
            return

        current_card = self.cards[self.card_idx]
        self.title(f'Etiquetando Carta {self.card_idx + 1}/{len(self.cards)} (ID: {current_card.cardId})')

        # Actualizar combobox de carta
        self.card_suit_combo.set(current_card.realSuit)
        self.card_figure_combo.set(current_card.realFigure)

        # Mostrar imagen de la carta completa
        if current_card.colorImage is not None and current_card.colorImage.size > 0:
            img_card_display = current_card.colorImage.copy()
            # Opcional: Dibujar el bounding box de la carta
            # cv2.rectangle(img_card_display, (current_card.boundingBox.x, current_card.boundingBox.y),
            #               (current_card.boundingBox.x + current_card.boundingBox.width, current_card.boundingBox.y + current_card.boundingBox.height),
            #               (0, 255, 0), 2)
            
            # Dibujar el contorno del motivo actualmente seleccionado en la imagen completa de la carta
            if current_card.motifs and self.motif_idx < len(current_card.motifs):
                current_motif = current_card.motifs[self.motif_idx]
                if current_motif.contour is not None and len(current_motif.contour) > 0:
                    cv2.drawContours(img_card_display, [current_motif.contour], -1, (255, 0, 255), 3) # Contorno del motivo en magenta
            
            img_card_display_rgb = cv2.cvtColor(img_card_display, cv2.COLOR_BGR2RGB)
            img_card_pil = Image.fromarray(img_card_display_rgb)
            img_card_pil = img_card_pil.resize((int(img_card_pil.width * (300/img_card_pil.height)), 300), Image.Resampling.LANCZOS) # Redimensionar para mostrar
            self.photo_card = ImageTk.PhotoImage(image=img_card_pil)
            self.card_image_label.config(image=self.photo_card)
        else:
            self.card_image_label.config(image=None)

        # Mostrar motivo individual
        if current_card.motifs and self.motif_idx < len(current_card.motifs):
            current_motif = current_card.motifs[self.motif_idx]
            self.motif_combo.set(current_motif.motifLabel)

            # Crear una imagen en blanco para dibujar el motivo
            if current_card.grayImage is not None and current_card.grayImage.size > 0:
                h, w = current_card.grayImage.shape[:2]
                img_display = np.zeros((h, w, 3), dtype=np.uint8) # Imagen en negro 3 canales

                if current_motif.contour is not None and len(current_motif.contour) > 0:
                    # Dibujar el contorno en la imagen en blanco (o sobre la imagen gris de la ROI)
                    # Para verlo claramente, dibujamos el contorno en una imagen blanca o negra.
                    # Podemos dibujar el contorno directamente en la máscara binaria del motivo o en una copia de la región de interés.
                    # Aquí vamos a crear una máscara del motivo y mostrarla para mayor claridad.
                    motif_mask = np.zeros_like(current_card.grayImage)
                    cv2.drawContours(motif_mask, [current_motif.contour], -1, 255, cv2.FILLED)
                    
                    # Convertir a 3 canales para mostrar en color si se desea, o simplemente mostrar la máscara.
                    # Aquí lo mostramos en blanco y negro, pero con el contorno de color para destacar
                    img_display_motif_colored = cv2.cvtColor(motif_mask, cv2.COLOR_GRAY2BGR)
                    cv2.drawContours(img_display_motif_colored, [current_motif.contour], -1, (0, 255, 0), 2) # Contorno en verde
                
                    img_motif_pil = Image.fromarray(img_display_motif_colored)
                    img_motif_pil = img_motif_pil.resize((200, 200), Image.Resampling.LANCZOS) # Redimensionar para mostrar
                    self.photo_motif = ImageTk.PhotoImage(image=img_motif_pil)
                    self.motif_image_label.config(image=self.photo_motif)
                else:
                    self.motif_image_label.config(image=None)
            else:
                self.motif_image_label.config(image=None)
        else:
            self.motif_combo.set('')
            self.motif_image_label.config(image=None)

        self.update_buttons()

    def update_buttons(self):
        # Habilitar/deshabilitar botones de navegación de motivos
        if not self.cards or not self.cards[self.card_idx].motifs:
            self.prevMotif_button.config(state=tk.DISABLED)
            self.nextMotif_button.config(state=tk.DISABLED)
            self.motif_combo.config(state=tk.DISABLED)
            self.apply_similar_button.config(state=tk.DISABLED)
        else:
            self.motif_combo.config(state="readonly")
            self.apply_similar_button.config(state=tk.NORMAL)
            if self.motif_idx == 0:
                self.prevMotif_button.config(state=tk.DISABLED)
            else:
                self.prevMotif_button.config(state=tk.NORMAL)
            
            if self.motif_idx >= len(self.cards[self.card_idx].motifs) - 1:
                self.nextMotif_button.config(state=tk.DISABLED)
            else:
                self.nextMotif_button.config(state=tk.NORMAL)
        
        # Habilitar/deshabilitar botones de navegación de cartas
        if self.card_idx == 0:
            self.prevCard_button.config(state=tk.DISABLED)
        else:
            self.prevCard_button.config(state=tk.NORMAL)

        if self.card_idx >= len(self.cards) - 1:
            self.nextCard_button.config(state=tk.DISABLED)
        else:
            self.nextCard_button.config(state=tk.NORMAL)

    def on_motif_label_selected(self, event):
        if self.cards and self.cards[self.card_idx].motifs:
            self.cards[self.card_idx].motifs[self.motif_idx].motifLabel = self.motif_combo.get()
            # No actualizamos la pantalla aquí, se hará al navegar o al guardar.

    def on_card_suit_selected(self, event):
        if self.cards:
            self.cards[self.card_idx].realSuit = self.card_suit_combo.get()

    def on_card_figure_selected(self, event):
        if self.cards:
            self.cards[self.card_idx].realFigure = self.card_figure_combo.get()

    def prev_motif(self):
        if self.motif_idx > 0:
            self.motif_idx -= 1
            self.update_display()

    def next_motif(self):
        if self.cards and self.motif_idx < len(self.cards[self.card_idx].motifs) - 1:
            self.motif_idx += 1
            self.update_display()
        elif self.card_idx < len(self.cards) - 1: # Si es el último motivo, ir a la siguiente carta
            self.next_card()

    def prev_card(self):
        if self.card_idx > 0:
            self.card_idx -= 1
            self.motif_idx = 0 # Reiniciar índice de motivo al cambiar de carta
            self.update_display()

    def next_card(self):
        if self.card_idx < len(self.cards) - 1:
            self.card_idx += 1
            self.motif_idx = 0 # Reiniciar índice de motivo al cambiar de carta
            self.update_display()
        else:
            messagebox.showinfo("Fin de las cartas", "Has llegado al final de la lista de cartas.")
            self.save_all_and_exit_prompt() # Preguntar si guardar al llegar al final

    def apply_to_similar_motifs(self):
        current_card = self.cards[self.card_idx]
        if not current_card.motifs or self.motif_idx >= len(current_card.motifs):
            messagebox.showinfo("Info", "No hay un motivo seleccionado o la carta no tiene motivos.")
            return

        selected_motif = current_card.motifs[self.motif_idx]
        selected_label = selected_motif.motifLabel

        if selected_label == 'i' or selected_label == 'Others':
            messagebox.showwarning("Advertencia", "Por favor, primero etiqueta el motivo actual antes de aplicar a similares.")
            return
        if selected_motif.features is None or len(selected_motif.features) == 0:
            messagebox.showwarning("Advertencia", "El motivo seleccionado no tiene características Hu Moments para comparar.")
            return

        labeled_count = 0
        for i, motif in enumerate(current_card.motifs):
            # Solo auto-etiquetar si el motivo no está ya etiquetado o está como 'Others' (para corregir errores)
            # y si tiene características Hu Moments para comparar.
            if (motif.motifLabel == 'i' or motif.motifLabel == 'Others') and \
               motif.features is not None and len(motif.features) > 0 and i != self.motif_idx:
                
                # Calcular la distancia euclidiana entre los Momentos de Hu
                # Asegurarse de que ambos arrays tienen la misma forma
                if selected_motif.features.shape == motif.features.shape:
                    distance = np.linalg.norm(selected_motif.features - motif.features)
                    
                    if distance < HU_MOMENT_SIMILARITY_THRESHOLD:
                        motif.motifLabel = selected_label
                        labeled_count += 1
                else:
                    warnings.warn(f"Advertencia: Diferente número de características Hu Moments para motivos {self.card_idx}-{self.motif_idx} y {self.card_idx}-{i}. Saltando comparación.")

        if labeled_count > 0:
            messagebox.showinfo("Éxito", f"Se aplicó la etiqueta '{selected_label}' a {labeled_count} motivos similares en esta carta.")
            self.update_display() # Actualizar la visualización de la carta para reflejar los cambios
        else:
            messagebox.showinfo("Info", "No se encontraron motivos similares no etiquetados para aplicar la etiqueta.")


    def save_current_card(self):
        if not self.cards:
            messagebox.showwarning("Advertencia", "No hay cartas para guardar.")
            return

        try:
            np.savez(filecard, Cartas=np.array(self.cards, dtype=object)) # Asegurarse de guardar como array de objetos
            messagebox.showinfo("Guardado", f"Carta {self.card_idx + 1} y todos los cambios guardados en {filecard}.")
            # Actualizar las etiquetas originales almacenadas
            for c_idx, card in enumerate(self.cards):
                for m_idx, motif in enumerate(card.motifs):
                    self.original_motif_labels[(c_idx, m_idx)] = motif.motifLabel
        except Exception as e:
            messagebox.showerror("Error de guardado", f"No se pudo guardar el archivo: {e}")

    def save_all_and_exit_prompt(self):
        # Comprobar si hay cambios sin guardar
        changes_made = False
        for c_idx, card in enumerate(self.cards):
            for m_idx, motif in enumerate(card.motifs):
                if self.original_motif_labels.get((c_idx, m_idx)) != motif.motifLabel:
                    changes_made = True
                    break
            if changes_made:
                break
        
        if changes_made:
            if messagebox.askyesno("Guardar y Salir", "¿Deseas guardar los cambios antes de salir?"):
                try:
                    np.savez(filecard, Cartas=np.array(self.cards, dtype=object))
                    messagebox.showinfo("Guardado", f"Todos los cambios guardados en {filecard}.")
                except Exception as e:
                    messagebox.showerror("Error de guardado", f"No se pudo guardar el archivo: {e}")
        self.destroy() # Cierra la aplicación

    def exit_app(self):
        self.save_all_and_exit_prompt() # Llama a la función para preguntar si guardar antes de salir

if __name__ == "__main__":
    app = App()
    app.mainloop()
