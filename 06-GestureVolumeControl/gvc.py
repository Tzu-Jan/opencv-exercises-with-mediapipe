import cv2
import mediapipe as mp
import time
import numpy as np

import osascript
import HandTrackingModule as htm
import math
import osascript  # for Mac volume control


widCam, heiCam = 1280, 960

cap = cv2.VideoCapture(1)
cap.set(3, widCam)
cap.set(4, heiCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)

minVol, maxVol = 0, 100
vol = 0
volBar = 400

while True:

    ret, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)
    if len(lmlist) != 0:
        # print(lmlist[4], lmlist[8])

        # position of thumb
        x1, y1 = lmlist[4][1], lmlist[4][2]
        # position of forefinger
        x2, y2 = lmlist[8][1], lmlist[8][2]

        cx, cy = (x1+x2)//2, (y1+y2)//2

        # thumb
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        # forefinger
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        # line to link thumb and forefinger
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        # center point
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        length = math.hypot(x2-x1, y2-y1)

        # transform length to vol
        vol = np.interp(length, [50, 300], [minVol, maxVol])
        volBar = np.interp(length, [50, 300], [400, 150])
        # set volume
        osascript.osascript("set volume output volume {}".format(vol))
        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

    # volume bar
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(vol)}%', (40, 450),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
