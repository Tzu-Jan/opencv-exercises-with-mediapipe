import cv2
import mediapipe as mp
import time

import poseModule as ps

pTime = 0
cTime = 0
cap = cv2.VideoCapture('05-Pose/videos/loco.mp4')
detector = ps.poseDetector()

size = (608, 1080)
output = cv2.VideoWriter('Pose/videos/YejisJaw.mp4', cv2.VideoWriter_fourcc(*'MP4V'),
                         10, size)

while True:
    ret, img = cap.read()
    img = detector.findPoses(img)
    lmList = detector.findPosition(img, draw=False)
    cv2.circle(img, ((lmList[9][1]+lmList[10][1])//2, (lmList[9][2]+lmList[10][2])//2),
               15, (255, 255, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    output.write(img)
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
output.release()

# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")
