import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import mediapipe as mp

# Inicializar o detector de face do MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Carregar o modelo VGG16 pré-treinado para classificação
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

# Função para processar frame e detectar/reconhecer faces
def process_frame(frame):
    # Converter BGR para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    
    if results.detections:
        for detection in results.detections:
            # Obter coordenadas da face detectada
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            width, height = int(bbox.width * w), int(bbox.height * h)
            
            # Extrair a região da face
            face_roi = frame[y:y+height, x:x+width]
            if face_roi.size == 0:
                continue
                
            # Preparar a face para classificação
            face_roi = cv2.resize(face_roi, (224, 224))
            face_array = image.img_to_array(face_roi)
            face_array = np.expand_dims(face_array, axis=0)
            face_array = preprocess_input(face_array)
            
            # Classificar a face
            features = model.predict(face_array)
            
            # Desenhar retângulo e informações
            cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {detection.score[0]:.2f}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

# Inicializar webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Processar frame
    output_frame = process_frame(frame)
    
    # Mostrar resultado
    cv2.imshow('Face Detection and Recognition', output_frame)
    
    # Pressionar 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
