import sys
from ultralytics import YOLO
import cv2
import numpy as np
import time

model = YOLO("best.pt", task="detect")
print("model loaded")

CONF_THRESH = 20


def obj_det(im_name, img, show, Ev3):
    results = model(img, show=show, imgsz=640)
    if show:
        cv2.waitKey()
    result = results[0]
    classes = result.boxes.cls.tolist()
    boxes = result.boxes.xyxy.tolist()
    """traffic signs {0:traffic_light, 1:stop, 2:speedlimit, 3:crosswalk, 4:redlight,  5:greenlight}"""
    for x in classes:
        print(x)
        if x == 4:
            return True
        elif x == 1:
            Ev3.write("s\n")
            time.sleep(3)
            return True
    return False


if __name__ == "__main__":
    while True:
        input()
        args = sys.argv[1:]
        print(obj_det(args[0], cv2.imread(args[0]), True, None))
