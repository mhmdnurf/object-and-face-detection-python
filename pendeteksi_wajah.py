import cv2 as cv

faceDetect = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
camera = cv.VideoCapture(0)

while True:
    ret, img = camera.read()
    img = cv.flip(img, 1)
    resize = cv.resize(img, (1366, 705))
    gray = cv.cvtColor(resize, cv.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        cv.rectangle(resize, (x, y), (x + w, y + h), (245, 245, 245), 2)
    cv.imshow("Pendeteksi Wajah", resize)
    if cv.waitKey(1) == 27:
        break
camera.release()
cv.destroyAllWindows()
