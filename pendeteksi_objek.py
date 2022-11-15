# Importing the OpenCV library.
import cv2 as cv

# The threshold for the confidence of the object detection.
thres = 0.45

# Using the default webcam.
cap = cv.VideoCapture(0)

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
net = cv.dnn_DetectionModel(weightsPath, configPath)

# Setting the input size of the model.
net.setInputSize(320, 320)

# Normalizing the input image.
net.setInputScale(1.0 / 127.5)


# This is the main loop of the program. It reads the image from the camera, flips it and detects the objects.
while True:
    success, img = cap.read()
    img = cv.flip(img, 1)
    resize = cv.resize(img, (1366, 705))
    classIds, confs, bbox = net.detect(resize, confThreshold=thres)
    print(classIds, bbox)

    # Drawing the bounding boxes around the detected objects.
    # Checking if the model has detected any objects.
    if len(classIds) != 0:
        # Drawing the bounding boxes around the detected objects.
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv.rectangle(resize, box, color=(153, 255, 255), thickness=1)
            # Drawing the name of the detected object on the image.
            cv.putText(
                resize,
                classNames[classId - 1].upper(),
                (box[0] + 10, box[1] + 30),
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (153, 255, 255),
                2,
            )

    # Showing the image.
    cv.imshow("Pendeteksi Objek", resize)
    
    # Checking if the user has pressed the escape key. If the user has pressed the escape key, the program will exit.
    if cv.waitKey(1) == 27:
        break
    #END
