
import cv2 as cv

faceDetect = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
camera = cv.VideoCapture(0)
camera.set(10,70)

while True: 
    jumlah=0
    ret, img = camera.read()
    img = cv.flip(img, 1)
    resize = cv.resize(img, (1366, 766)) 
    faces = faceDetect.detectMultiScale(resize)
    for (x, y, w, h) in faces:
        jumlah = jumlah + 1

        cv.rectangle(resize, (x, y), (x + w, y + h), (153, 255 , 255), 1)
        cv.putText(resize, 'Wajah', (x,y-10), cv.FONT_HERSHEY_COMPLEX, 1, (153,255,255),2) 
    cv.putText(resize,"Wajah terdeteksi : "+ str(jumlah),(10,30),cv.FONT_HERSHEY_COMPLEX,1,(153,255,255),2)
    cv.imshow("Pendeteksi Wajah", resize)
    if cv.waitKey(1) == 27:
        break 

