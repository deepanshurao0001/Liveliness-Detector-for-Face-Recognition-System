import cv2
from cvzone.FaceDetectionModule import FaceDetector
import cvzone
from time import time, sleep  # Import the sleep function from the time module

#   Percentage values for extra offset than the face detector.
offsetPercentW = 10
offsetPercentH = 20

#   Confidence value for detecting any face.
confidence = 0.8
save = True
blurThreshold = 50

#   Height and width of camera frame
camWidth = 640
camHeight = 480

#   Add the path of folder for saving data
outputFolderPath = 'Dataset/DataCollect'

#   Set ID for classifying weather the saved images are real or fake
classID = 1     #   0 is fake and 1 is real.

#   Parameter for debugging the files that are getting saved
debug = False

#   The desired FPS at which we want to save images.
desired_fps = 15  

cap = cv2.VideoCapture(1)
cap.set(3, camWidth)
cap.set(4, camHeight)
detector = FaceDetector()

while True:
    start_time = time()  # Recording the start time of processing the frame

    success, img = cap.read()
    imgOut = img.copy()
    img, bboxes = detector.findFaces(img, draw=False)

    listBlur = []
    listInfo = []

    if bboxes:
        #   Elements of 'bbox' - "id", "bbox", "score", "center".
        for bbox in bboxes:
            x, y, w, h = bbox["bbox"]
            score = bbox["score"][0]

            if score > confidence:

                #   Enlarging the face detector box by adding offsets, so that it covers entire face.

                offsetw = (offsetPercentW / 100) * w
                x = int(x - offsetw)
                w = int(w + offsetw * 2)

                offseth = (offsetPercentH / 100) * w
                y = int(y - offseth * 3)
                h = int(h + offseth * 3.5)

                #   Making sure that the face remains in the display window.

                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0

                #   Checking for the blurry or unclear images.

                imgFace = img[y:y + h, x:x + w]
                cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                if blurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)

                #   Normalizing the values.
                ih, iw, _ = img.shape
                xc, yc = x + w / 2, y + h / 2

                xcn = round(xc / iw, 6)
                ycn = round(yc / ih, 6)
                wn = round(w / iw, 6)
                hn = round(h / ih, 6)

                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1: hn = 1

                #   Below is the format which is required by YOLO for the dataset

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                #   Drawing the boxes on the display screen.

                cv2.rectangle(imgOut, (x, y, w, h), (0, 255, 0), 3)
                cvzone.putTextRect(imgOut, f'score: {int(score * 100)}% Blur: {blurValue}', (x, y - 20),
                                   scale=1.5, thickness=2)

                if debug:
                    cv2.rectangle(img, (x, y, w, h), (0, 255, 0), 3)
                    cvzone.putTextRect(img, f'score: {int(score * 100)}% Blur: {blurValue}', (x, y - 20),
                                       scale=1.5, thickness=2)

        #   Saving Files for Dataset

        if save:
            if all(listBlur) and listBlur != []:
                #   Saving images
                timeNow = time()
                timeNow = str(timeNow).split('.')
                timeNow = timeNow[0] + timeNow[1]
                cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)

                #   Saving text files with information about images
                for info in listInfo:
                    f = open(f"{outputFolderPath}/{timeNow}.txt", 'a')
                    f.write(info)
                    f.close()

    cv2.imshow("Image", imgOut)

    # Calculating the time taken for processing the frame.
    processing_time = time() - start_time

    # Calculating the delay required to achieve the desired FPS.
    delay_time = 1 / desired_fps - processing_time

    # If processing took less time than the desired frame rate, introduce a delay.
    if delay_time > 0:
        sleep(delay_time)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
