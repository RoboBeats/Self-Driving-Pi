import sys
from ultralytics import YOLO
import cv2
model = YOLO("best.pt")
print("model loaded")

CONF_THRESH = 20

def obj_det(im_name, show):
    results = model(im_name, show=show)
    if show:
        cv2.waitKey()
    result = results[0]
    classes = result.boxes.cls.tolist()
    boxes = result.boxes.xyxy.tolist()
    final_results = []
    for i in range(len(classes)):
        final_results.append([int(classes[i]), [int(x) for x in boxes[i]]])
    return final_results

if __name__ == "__main__":
    args = sys.argv[1:]
    print(obj_det(args))