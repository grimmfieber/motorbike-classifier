import cv2 as cv
import numpy as np
import logging as log
import pyautogui
import imutils
import os

net = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')

classes = []

with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

cap = cv.VideoCapture('test.mp4')
#img = cv.imread('image.jpg')

try:

    # creating a folder named data
    if not os.path.exists('data'):
        os.makedirs('data')

    # if not created then raise error
except OSError:
    print('Error: Creating directory of data')

goruldu = False
currentframe = 0
frame_per_second = cap.get(cv.CAP_PROP_FPS) 
frames_captured = 0
step = 10
frames_count = 3

font = cv.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x=int(center_x - w/2)
                y=int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    if len(indexes)>0:
        for i in indexes.flatten():
            for j in indexes.flatten():
                label = str(classes[class_ids[i]])
                if label=="motorbike":
                    if goruldu == 0:
                        log.warning('Bir motor görüldü')
                        goruldu = True
                        # if video is still left continue creating images
                        name = './data/frame' + str(currentframe) + '.jpg'
                        print('Creating...' + name)
                        # writing the extracted images
                        cv.imwrite(name, img)
                        # increasing counter so that it will
                        # show how many frames are created
                        currentframe += 1         
                    x, y, w, h = boxes[i]
                    confidence = str(round(confidences[i], 2))
                    color = colors[i]
                    cv.rectangle(img, (x,y), (x+w, y+h), color, 2)
                    cv.putText(img, label + " " + confidence, (x, y+20), font, 2, (0, 0, 0), 2)
                    #pyautogui.screenshot("straight_to_disk.png")
                    
    cv.imshow('Image', img)
    key = cv.waitKey(1)
    if key==27:
        break

cap.release()
cv.destroyAllWindows()