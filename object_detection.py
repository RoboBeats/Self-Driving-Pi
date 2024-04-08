import sys
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("best.onnx", task="detect")
print("model loaded")

CONF_THRESH = 20

def obj_det(im_name, img, show):
    results = model(im_name, show=show, imgsz=320)
    if show:
        cv2.waitKey()
    result = results[0]
    classes = result.boxes.cls.tolist()
    boxes = result.boxes.xyxy.tolist()
    """traffic signs {0:traffic_light, 1:top, 2:speedlimit, 3:crosswalk}"""
    stop = False
    for i in range(len(classes)):
#         if int(classes[i]) == 0:
#             box = boxes[i]
#             print(box)
#             traffic_light = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
#             if show:
#                 cv2.imshow("trafficlight", traffic_light)
#                 cv2.waitKey()
#             hsv_light = cv2.cvtColor(traffic_light, cv2.COLOR_BGR2HSV)
#             gray = cv2.cvtColor(hsv_light, cv2.COLOR_BGR2GRAY)
#             blur = cv2.GaussianBlur(gray, (5, 5), 0)
#             circles = cv2.HoughCircles(
#                 blur, cv2.HOUGH_GRADIENT, 1, minDist=20, param1=10, param2=20, minRadius=25, maxRadius=0)
#             print(circles)
#             if show:
#                 for circle in circles[0]:
#                     print(circle[:2])
#                     cv2.circle(traffic_light, [int(x) for x in circle[:2]], int(circle[2]), (120, 120, 120), 2)
#                     cv2.imshow("trafficlight", traffic_light)
#                     cv2.waitKey()
         
        if int(classes[i]) in (0, 1):
            stop = True
    return stop

if __name__ == "__main__":
    args = sys.argv[1:]
    print(obj_det(args[0], True))