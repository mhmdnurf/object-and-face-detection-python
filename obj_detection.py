import cv2


thres = 0.45  # Threshold to detect object
# img = cv2.imread("human_with_cat.jpg", cv2.IMREAD_UNCHANGED)
# width = 1280
# height = 720
# dim = (width, height)

# res = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
url = "https://192.168.1.7:8080/video"
cap = cv2.VideoCapture(url)
# cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 70)

classNames = []
classFile = "coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray)
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    print(classIds, bbox)

    if len(classIds) != 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.imshow("Face", img)
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
            cv2.putText(
                img,
                classNames[classId - 1].upper(),
                (box[0] + 10, box[1] + 30),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                img,
                str(round(confidence * 100, 2)),
                (box[0] + 200, box[1] + 30),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 255, 0),
                2,
            )

    cv2.imshow("Output", img)
    if cv2.waitKey(1) == 27:
        break
