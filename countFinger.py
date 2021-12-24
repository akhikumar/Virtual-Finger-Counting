from typing import overload
import cv2
import time
import os
from cvzone import HandTrackingModule
# import HandTrackingModule as htm

wCam , hCam = 640 , 480

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)

folderPath = "fingerImages"
myList = os.listdir(folderPath)
print(myList)
overLayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overLayList.append(image)

print(len(overLayList))
pTime = 0

detector = HandTrackingModule.HandDetector(detectionCon=0.75)
tipIds = [4,8,12,16,20]


while True:

    success,img = cap.read()
    hands,img = detector.findHands(img)

    if hands:
        hand1 = hands[0]
        lmlist1 = hand1["lmList"]
        # print(len(lmlist1),lmlist1)
        # print("==============",lmlist1[tipIds[0]][1])
        # print("###########",lmlist1[tipIds[0]])
        # print("**********",lmlist1[tipIds[0]-1][1])
        # print("$$$$$$$$$$",lmlist1[tipIds[0]-1])
        # print("@@@@@@@@",lmlist1[tipIds[1]][1])
        # print("&&&&&&&&",lmlist1[tipIds[1]-2][1])
    
        if len(lmlist1) != 0:
            fingers = []

            # Thumb
            if lmlist1[tipIds[0]][0] > lmlist1[tipIds[0]-1][0]:
                fingers.append(1)
            else:
                fingers.append(0)
            
            # 4 left fingers
            for id in range(1,5):
                if lmlist1[tipIds[id]][1] < lmlist1[tipIds[id]-2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            # print(fingers)
            totalFingers = fingers.count(1)
            print(totalFingers)

            h,w,c = overLayList[totalFingers - 1].shape
            img[0:h, 0:w] = overLayList[totalFingers - 1]

            cv2.rectangle(img,(20,255),(178,425),(0,255,0),cv2.FILLED)
            cv2.putText(img,str(totalFingers),(45,375),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),25)
        
        cTime = time.time()
        fps = 1 / (cTime -pTime)
        pTime = cTime

        cv2.putText(img,f'FPS : {int(fps)}',(400,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)

        





    cv2.imshow("results",img)
    cv2.waitKey(1)
