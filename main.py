import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace 

img = cv2.imread('happy.jpg')
plt.imshow(img)

prediction=DeepFace.analyze(img)
print(prediction[0]['dominant_emotion'])

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')