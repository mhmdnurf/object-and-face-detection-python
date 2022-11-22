import cv2 as cv #untuk mengimport library dari OpenCV

thres = 0.5 #ambang batas untuk membatasi program dalam mendeteksi objek

cap = cv.VideoCapture(0) #untuk menentukan program agar menggunakan webcam default laptop

# cap.set(10,50)

classNames = [] #array kosong yang nantinya diisi oleh nama nama objek.

classFile = "coco.names" #untuk mengisi file yang terdapat nama nama objek yang dapat dideteksi oleh program.

with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n") #untuk membaca file dan diubah kedalam bentuk list.

configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt" #path atau jalur untuk file model konfigurasinya.

weightsPath = "frozen_inference_graph.pb" #path atau jalur untuk file weight konfigurasinya. 

net = cv.dnn_DetectionModel(weightsPath, configPath) #untuk memproses dan mendeteksi file konfigurasinya.

net.setInputSize(320, 320) #untuk mengatur lebar dan tinggi dari pendeteksi nya.

#Normalizing the input image.
net.setInputScale(1 / 127.5)

while True: #jika kondisi benar atau sesuai maka akan dijalankan codingan dibawahnya.
    
    ret, img = cap.read() #untuk membaca camera yang akan ditampilkan ke layar
    
    img = cv.flip(img, 3) #untuk membalikkan tampilan dari camera menjadi mirror seperti kamera selfie, nilai 0 untuk membalik sumbu-x, nilai 1 untuk membalik sumbu-y.

    resize = cv.resize(img, (640, 480)) #untuk mengatur resolusi yang ditampilkan pada camera, value pertama untuk mengatur lebar, value kedua untuk mengatur tinggi.


    classIds, confs, bbox = net.detect(resize, confThreshold=thres) #untuk mendeteksi objek yang tampil di kamera dan confThreshold merupakan ambang batas yang terdapat pada variabel thres.

    print(classIds, bbox) #untuk mencetak/merekam variabel classIds dan bbox yang terdapat pada codingan di atas.

    if len(classIds) != 0: #len digunakan untuk menghitung jumlah dari objek classIds,jika nilai != 0, maka for looping dibawahnya akan dijalankan.

        # Drawing the bounding boxes around the detected objects.
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox): #

            cv.rectangle(resize, box, (153, 255, 255),3) #untuk membuat persegi di daerah wajah yang terdeteksi dimana value angka diisi oleh format GBR untuk warna dari perseginya. Dan nilai 1 setelah value GBR berfungsi untuk mengatur ketebalan dari garis atau sisi perseginya.
    
            cv.putText(resize,classNames[classId - 1].upper(), (box[0] + 10,box[1] + 30),cv.FONT_HERSHEY_COMPLEX,1,(153, 255, 255),2) #untuk menampilkan tulisan objek yang terdeteksi 

    cv.imshow("Pendeteksi Objek", resize) #untuk menampilkan camera

    if cv.waitKey(1) == 27: #waitKey dengan nilai 1 untuk memberikan delay sebesar 1ms terhadap gerakan yang dilakukan didepan camera,dan akan berhenti ketika tombol ESC ditekan. nilai 27 = ESC diambil dari ASCII yaitu American Standard Code for Information Interchange dimana nilai 27 = ESC.
        break #setelah ESC ditekan, looping akan berhenti dan camera akan mati.
