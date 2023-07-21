from ultralytics import YOLO
import cv2
import cvzone
import math


cap = cv2.VideoCapture("../Videos/1.avi") #for downloaded videos

model = YOLO("../YOLO-Weights/yolov8l.pt")
names = model.names

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            """cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)"""

            w, h = x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))
            #confidence level
            conf = math.ceil((box.conf[0]*100))/100
            #cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(35,y1)))

            #class name
            cls = box.cls[0]
            cvzone.putTextRect(img, f'{names[int(cls)]} {conf}', (max(0, x1), max(35, y1)),scale=1, thickness=1)
    cv2.imshow("Image",img)
    cv2.waitKey(1)