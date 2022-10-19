import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("04-FaceMesh/videos/yuji.mp4")
# cap = cv2.VideoCapture("04-FaceMesh/videos/Yejin.mp4")
pTime = 0


mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
mpMesh = mp.solutions.mediapipe.python.solutions.face_mesh  # mp.solutions.pose
meshs = mpMesh.FaceMesh(max_num_faces=2)
drawSpecPo = mpDraw.DrawingSpec(
    color=(0, 0, 255), thickness=3, circle_radius=1)
drawSpecLine = mpDraw.DrawingSpec(
    color=(0, 255, 0), thickness=3, circle_radius=2)

while True:
    ret, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = meshs.process(imgRGB)
    if results.multi_face_landmarks:
        for facelms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(
                img, facelms, mpMesh.FACEMESH_CONTOURS, drawSpecPo, drawSpecLine)

            for id, lm in enumerate(facelms.landmark):
                # print(lm)
                # convert lm to pixels
                ih, iw, ic = img.shape
                x, y = int(lm.x*iw), int(lm.y*ih)
                # print(id, x, y)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.rectangle(img, (15, 80), (210, 20), (255, 255, 0),  cv2.FILLED)
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("result", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
