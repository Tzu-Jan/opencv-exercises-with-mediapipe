import cv2
import mediapipe as mp
import time


class poseDetector():
    def __init__(self, mode=False, upbody=False,
                 smooth=True, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.upbody = upbody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
        self.mpPose = mp.solutions.mediapipe.python.solutions.pose
        self.poses = self.mpPose.Pose()
        # self.mode, self.upbody, self.smooth, self.detectionCon, self.trackCon)

    def findPoses(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.poses.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS, self.mpDraw.DrawingSpec(
                                               color=(0, 255, 0), thickness=2, circle_radius=2),  # color of landmarks
                                           self.mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2))  # color of coonrtion lines

        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 255, 0), cv2.FILLED)
        return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture("Pose/videos/loco.mp4")
    detector = poseDetector()

    while cap.isOpened():
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

        cv2.imshow("Result", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
