from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("../Videos/people.mp4") #for downloaded videos

model = YOLO("../YOLO-Weights/yolov8l.pt")
names = model.names

mask = cv2.imread("mask.png")

#sort/tracking
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)
limitsUp = [103,161,296,161]
limitsDown= [527,489,735,489]
upCount=[]
downCount=[]
while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)

    imgGraphics = cv2.imread("graphics.png",cv2.IMREAD_UNCHANGED)#in the while loop to keep the graphics the same
    img=cvzone.overlayPNG(img,imgGraphics,(730,260))

    results = model(imgRegion, stream=True)

    detections = np.empty((0,5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            #bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2-x1,y2-y1

            #confidence level
            conf = math.ceil((box.conf[0]*100))/100

            #class name
            cls = int(box.cls[0])
            currentClass = names[cls]
            print(currentClass)


            if currentClass == "person" and (conf > 0.3):
                #cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                #cvzone.putTextRect(img, f'{names[cls]} {conf}', (max(0, x1), max(35, y1)),scale=1, thickness=1,offset=3)

                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    """cv2.line(img,(limitsUp[0],limitsUp[1]),(limitsUp[2],limitsUp[3]),(0,0,255),5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)"""
    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), 5)

    for results in resultsTracker:
        x1,y1,x2,y2,id = results
        print(results)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1,y1,w,h), l=9 , rt=2 , colorR=(255,0,255))
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)
        cx,cy = x1+w//2 , y1+h//2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

        if limitsUp[0] <cx < limitsUp[2] and limitsUp[1]-15 <cy< limitsUp[3]+15:
            if upCount.count(id)==0:
                upCount.append(int(id))
                #cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)

        if limitsDown[0] <cx < limitsDown[2] and limitsDown[1]-15 <cy< limitsDown[3]+15:
            if downCount.count(id)==0:
                downCount.append(int(id))
                #cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)




        #cvzone.putTextRect(img, f'Count :{len(totalCount)}', (50,50))
        cv2.putText(img,str(len(upCount)),(929,345),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),8)
        cv2.putText(img, str(len(downCount)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
    cv2.imshow("Image",img)
    cv2.waitKey(1)