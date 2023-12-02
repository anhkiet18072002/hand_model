import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import paho.mqtt.client as paho

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")

offset = 20
imgSize = 300
pub_topic = "HieuTeo"

client = paho.Client()

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(pub_topic)

def on_publish(client, userdata, mid):
    print("Message {} Published to MQTT topic".format(mid))

client.on_connect = on_connect
client.on_publish = on_publish

client.connect('broker.mqttdashboard.com', 1883)
max = 0
while True:
    ret, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        if x - offset >= 0 and y - offset >= 0 and x + w + offset <= img.shape[1] and y + h + offset <= img.shape[0]:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print("Index:", index)
            if max !=index:
                max = index
                (rc, mid) = client.publish(pub_topic, str(index), qos=1)
                if rc == paho.MQTT_ERR_SUCCESS:
                    print("Message successfully sent")
                else:
                    print("Message could not be sent")
        else:
            print("Invalid region of interest. Skipping this frame.")
    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break