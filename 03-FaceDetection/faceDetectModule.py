import cv2
import mediapipe as mp
import time


class faceDetector():
    def __init__(self,  minDetect=0.5,):

        self.minDetect = minDetect

        self.mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
        self.mpFacedetect = mp.solutions.mediapipe.python.solutions.face_detection
        self.facedetect = self.mpFacedetect.FaceDetection(0.7)

    def findFaces(self, img,  draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.facedetect.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxc = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxc.xmin * iw), int(bboxc.ymin * ih),\
                int(bboxc.width * iw), int(bboxc.height * ih)
            bboxs.append([id, bbox, detection.score])
            if draw:
                self.fancyDraw(img, bbox)

                cv2.putText(img, f'Face: {int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)
        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=10):
        x, y, w, h = bbox
        x1, y1 = x + w, y+h

        cv2.rectangle(img, bbox, (255, 0, 255))

        # top left
        cv2.line(img, (x, y), (x+l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 255), t)

        # top right
        cv2.line(img, (x1, y), (x1-l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y+l), (255, 0, 255), t)

        # btm right
        cv2.line(img, (x1, y1), (x1-l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1-l), (255, 0, 255), t)

        # btm left
        cv2.line(img, (x, y1), (x+l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1-l), (255, 0, 255), t)

        return img


def main():

    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture("FaceDetection/videos/mamamoo.mp4")
    detector = faceDetector(0.4)

    while cap.isOpened():
        ret, img = cap.read()
        img, bbooxs = detector.findFaces(img)
        # lmList = detector.findPosition(img)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Result", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
