import cv2
import numpy as np
from app.model_loader import load_fruit_model
import json

# Cargar el modelo y las etiquetas
model_path = 'model/modelo_frutas.h5'
labels_path = 'model/etiquetas_frutas.json' # Si tienes un archivo de etiquetas

fruit_model = load_fruit_model(model_path)

# Si tienes un archivo de etiquetas, cárgalo:
try:
    with open(labels_path, 'r') as f:
        fruits = json.load(f)
    reverse_fruits = {v: k for k, v in fruits.items()}
except FileNotFoundError:
    print("Archivo de etiquetas no encontrado.  Las predicciones serán numéricas.")
    reverse_fruits = None

def preprocess_frame(frame):
    # Redimensiona la imagen a las dimensiones que espera tu modelo (ej. 50x50)
    resized_frame = cv2.resize(frame, (50, 50))
    # Normaliza los valores de los píxeles (si tu modelo fue entrenado con datos normalizados)
    normalized_frame = resized_frame / 255.0
    # Expande las dimensiones para que coincidan con la entrada del modelo
    expanded_frame = np.expand_dims(normalized_frame, axis=0)
    return expanded_frame

def predict_fruit(frame):
    processed_frame = preprocess_frame(frame)
    prediction = fruit_model.predict(processed_frame)
    predicted_class_index = np.argmax(prediction)

    if reverse_fruits:
        predicted_label = reverse_fruits.get(predicted_class_index, "Desconocido")
    else:
        predicted_label = str(predicted_class_index) # Si no hay etiquetas, muestra el índice
    probability = prediction[0][predicted_class_index]
    return predicted_label, probability

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # 0 es la cámara predeterminada

    if not cap.isOpened():
        print("No se pudo abrir la cámara")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el fotograma")
            break

        label, probability = predict_fruit(frame)
        text = f"{label}: {probability:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Detección de Frutas", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): # Presiona 'q' para salir
            break

    cap.release()
    cv2.destroyAllWindows()
