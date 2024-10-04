import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

# Función para cargar una imagen
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        lbl_image.config(image=img_tk)
        lbl_image.image = img_tk
        global img_path
        img_path = file_path

# Función para contar dedos
def detect_fingers(image_path):
    image = cv2.imread(image_path)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            fingers_up = count_fingers_up(hand_landmarks)
            return fingers_up
        else:
            return 0

def count_fingers_up(hand_landmarks):
    fingers_up = 0
    landmarks = hand_landmarks.landmark

    if landmarks[4].x > landmarks[2].x:  # Pulgar
        fingers_up += 1
    if landmarks[8].y < landmarks[6].y:  # Índice
        fingers_up += 1
    if landmarks[12].y < landmarks[10].y:  # Medio
        fingers_up += 1
    if landmarks[16].y < landmarks[14].y:  # Anular
        fingers_up += 1
    if landmarks[20].y < landmarks[18].y:  # Meñique
        fingers_up += 1

    return fingers_up

# Función para clasificar la mano
def classify_hand(image_path):
    fingers_up = detect_fingers(image_path)
    if fingers_up == 5:
        return "La mano tiene 5 dedos."
    else:
        return f"La mano tiene {fingers_up} dedos."

# Función para analizar la imagen
def analyze_image():
    if img_path:
        result = classify_hand(img_path)
        lbl_result.config(text=result)

# Interfaz gráfica con Tkinter
root = tk.Tk()
root.title("Detección de Dedos")

lbl_image = tk.Label(root)
lbl_image.pack()

btn_load = tk.Button(root, text="Cargar Imagen", command=load_image)
btn_load.pack()

btn_analyze = tk.Button(root, text="Analizar Imagen", command=analyze_image)
btn_analyze.pack()

lbl_result = tk.Label(root, text="")
lbl_result.pack()

root.mainloop()
