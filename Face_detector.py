
import cv2
from cvzone.FaceDetectionModule import FaceDetector

cap = cv2.VideoCapture(0)
detector = FaceDetector()

while True:
    success, img = cap.read()
    img, bboxes = detector.findFaces(img)

    if bboxes:
        center = bboxes[0]["center"]
        cv2.circle(img, center, 5, (255,0,255), cv2.FILLED)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
