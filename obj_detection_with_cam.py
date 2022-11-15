# Importing the OpenCV library.
import cv2

# The threshold for the confidence of the object detection.
thres = 0.45

# Detecting the face.
faceDetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Use mobile camera as a webcam
# url = "https://192.168.7.10:8080/video"
# cap = cv2.VideoCapture(url)

# Using the default webcam.
cap = cv2.VideoCapture(0)

# Setting the resolution of the camera.
cap.set(3, 1366)

# Setting the height of the camera.
cap.set(4, 768)

# Setting the brightness of the camera.
cap.set(10, 70)

# Creating an empty list.
classNames = []
# A file that contains the names of the objects that the model can detect.
classFile = "coco.names"
# Reading the file and splitting it into a list.
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# The path to the configuration file of the model.
configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

# The path to the weights file of the model.
weightsPath = "frozen_inference_graph.pb"

# Loading the model.
net = cv2.dnn_DetectionModel(weightsPath, configPath)

# Setting the input size of the model.
net.setInputSize(320, 320)

# Normalizing the input image.
net.setInputScale(1.0 / 127.5)

# Subtracting the mean from the input image.
net.setInputMean((127.5, 127.5, 127.5))

# Swapping the red and blue channels.
net.setInputSwapRB(True)

# This is the main loop of the program. It reads the image from the camera, flips it, converts it to
# grayscale, detects the faces, and detects the objects.
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray)
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    print(classIds, bbox)

    # Drawing the bounding boxes around the detected objects.
    # Checking if the model has detected any objects.
    if len(classIds) != 0:
        # Drawing the bounding boxes around the detected objects.
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 0, 255), thickness=2)
            # Drawing the name of the detected object on the image.
            cv2.putText(
                img,
                classNames[classId - 1].upper(),
                (box[0] + 10, box[1] + 30),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            # Drawing the confidence of the object detection on the image.
            cv2.putText(
                img,
                str(round(confidence * 100, 2)),
                (box[0] + 200, box[1] + 30),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (0, 0, 255),
                2,
            )

    # Showing the image.
    cv2.imshow("Output", img)
    # Checking if the user has pressed the escape key. If the user has pressed the escape key, the
    # program will exit.
    if cv2.waitKey(1) == 27:
        break

