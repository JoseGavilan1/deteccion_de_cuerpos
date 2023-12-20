import cv2
import argparse
from persondetection import DetectorAPI

def open_cam():
    args = argsParser()
    video = cv2.VideoCapture(0)
    odapi = DetectorAPI()
    threshold = 0.9

    while True:
        check, frame = video.read()
        img = cv2.resize(frame, (800, 600))
        boxes, scores, classes, num = odapi.processFrame(img)
        person = 0
        acc = 0

        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                person += 1
                cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
                cv2.putText(img, f'P{person, round(scores[i], 2)}', (box[1] - 30, box[0] - 8),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
                acc += scores[i]
                
        print(person)
        cv2.imshow("Deteccion de cuerpos", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def argsParser():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-c", "--camera", default=False, help="Set true if you want to use the camera.")
    args = vars(arg_parse.parse_args())
    return args

open_cam()