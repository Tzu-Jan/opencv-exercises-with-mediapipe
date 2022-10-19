import cv2
import mediapipe as mp
import time


class faceMeshCreator():
    def __init__(self, mode=False, maxFaces=2, detectCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxFaces = maxFaces
        self.detectCon = detectCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.mediapipe.python.solutions.drawing_utils
        self.mpMesh = mp.solutions.mediapipe.python.solutions.face_mesh  # mp.solutions.pose
        self.meshs = self.mpMesh.FaceMesh(max_num_faces=4)
        self.drawSpecPo = self.mpDraw.DrawingSpec(
            color=(0, 255, 0), thickness=1, circle_radius=1)
        self.drawSpecLine = self.mpDraw.DrawingSpec(
            color=(0, 255, 0), thickness=1, circle_radius=1)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.meshs.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:

            for facelms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, facelms, self.mpMesh.FACEMESH_CONTOURS, self.drawSpecPo, self.drawSpecLine)
                face = []
                for id, lm in enumerate(facelms.landmark):
                    # convert lm to pixels
                    ih, iw, ic = img.shape
                    x, y = int(lm.x*iw), int(lm.y*ih)
                    # cv2.putText(img, f'FPS: {int(id)}', (x, y),
                    #             cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 0, 255), 1)

                    # print(id, x, y)
                    face.append([x, y])
                faces.append(face)
        return img, faces

    def find():
        pass


def main():
    # cap = cv2.VideoCapture("FaceMesh/videos/yuji.mp4")
    cap = cv2.VideoCapture("FaceMesh/videos/Yejin.mp4")
    # cap = cv2.VideoCapture("FaceDetection/videos/mamamoo.mp4")
    pTime = 0
    detector = faceMeshCreator()

    while True:
        ret, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        # if len(faces) != 0:
        #     print(len(faces))

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.rectangle(img, (15, 80), (210, 20), (255, 255, 0),  cv2.FILLED)
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("result", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
