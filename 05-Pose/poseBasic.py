import cv2
import mediapipe as mp
import time

mpPose = mp.solutions.mediapipe.python.solutions.pose  # mp.solutions.pose
poses = mpPose.Pose()
mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils


pTime = 0
cTime = 0

cap = cv2.VideoCapture("Pose/videos/loco.mp4")

while cap.isOpened():
    ret, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = poses.process(imgRGB)
    # print(results.pose_landmarks)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks,
                              mpPose.POSE_CONNECTIONS, mpDraw.DrawingSpec(
                                  color=(245, 255, 66), thickness=2, circle_radius=2),  # color of coonetion lines
                              mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2))  # color of landmarks
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            # print(id, cx, cy)

            cv2.circle(img, (cx, cy), 5, (255, 255, 0), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("Result", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
