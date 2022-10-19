import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("faceDetection/videos/mamamoo.mp4")


pTime = 0
cTime = 0

mpFacedetect = mp.solutions.mediapipe.python.solutions.face_detection
facedetect = mpFacedetect.FaceDetection(0.7)
mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils

while cap.isOpened():
    ret, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = facedetect.process(imgRGB)
    # print(results.detections)

    if results.detections:
        for id, detection in enumerate(results.detections):
            # print(id, detection)
            # if detection.score[0] > 0.7:
            #     mpDraw.draw_detection(img, detection, mpDraw.DrawingSpec(
            #         color=(255, 0, 255), thickness=2), mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2))
            # print(detection.location_data.relative_bounding_box)
            bboxc = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxc.xmin * iw), int(bboxc.ymin * ih),\
                int(bboxc.width * iw), int(bboxc.height * ih)
            cv2.rectangle(img, bbox, (255, 0, 255))
            cv2.putText(img, f'Face: {int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {str(int(fps))}', (10, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
