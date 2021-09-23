import cv2
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

# Load model
final_model = tf.keras.models.load_model('checkpoint/best_model.h5') # Load model from checkpoint directory
label_to_text = {0:'Anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'Sadness', 5: 'Surprise', 6: 'neutral'}

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)    

    # Crop face and adjust resolution to model
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
    cv2.imshow('img2', face)
    
    # Reshape image to use in model
    face_to_predict = face.reshape(48,48,1)

    # Predict emotion
    predicted_label = final_model.predict(tf.expand_dims(face_to_predict, 0)).argmax()
    emotion = label_to_text[predicted_label]
    print(f'Emotion: {emotion}')

    # Display
    cv2.putText(img, f'Emotion: {emotion}', (x,y-1), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('img', img)
    
    
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
    
# Release the VideoCapture object
cap.release()