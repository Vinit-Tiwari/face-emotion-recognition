# import cv2
# import matplotlib.pyplot as plt
# from deepface import DeepFace

# img = cv2.imread('happy.jpg')
# plt.imshow(img)

# prediction = DeepFace.analyze(img)
# print(prediction[0]['dominant_emotion'])

# faceCascade = cv2.CascadeClassifier(
#     cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# faces = faceCascade.detectMultiScale(gray, 1.1, 4)

# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# font = cv2.FONT_HERSHEY_PLAIN
# cv2.putText(img, prediction[0]['dominant_emotion'],
#             (0, 50), font, 1, (0, 0, 255), 2, cv2.LINE_4)


import cv2
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    raise IOError("Cannot open webcam!")

while True:
    ret, frame = cap.read()
    result = DeepFace.analyze(
        frame, actions=['emotion'], enforce_detection=False)
    print(result)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(frame,  result[0]['dominant_emotion'],
                (0, 50), font, 1, (0, 0, 255), 2, cv2.LINE_4)

    cv2.imshow('Original video', frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
