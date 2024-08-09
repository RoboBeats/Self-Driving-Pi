import cv2
import sys
import numpy as np
import math
from lane_func import *
from merge_linesv2 import merge
import os
from copy import deepcopy

MAX_MERGE_DIST, MAX_MERGE_ANGLE = 100, 2
# for a piar of lanes
MIN_TOP, MAX_TOP = 300, 800
MIN_BOT, MAX_BOT = 400, 1000
# for single lane:
ANG_TOL = 20
prev_right = []
prev_left = []
DISPLACEMENT_TOL = 50
HEADING_DISP = 5

frame_num = [0]

def get_lanes(original_image, img_name, prev_heading, prev_left=prev_left, prev_right=prev_right, show=(__name__=="__main__")):
    # Load image, delete top portion, grayscale
    image = delete_top(original_image)
    #print(image.shape)

    mask = cv2.inRange(image, np.array([-1, -1, -1]), np.array([130, 130, 130]))
    kernel = np.ones((2, 2), np.uint8)
    erode = cv2.erode(mask, kernel)

    if show:
        cv2.imshow('original',original_image)
        cv2.waitKey()
#         cv2.destroyAllWindows()
    
    #ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    if show:
        cv2.imshow('erode', erode)
        cv2.waitKey()

    raw_hough_img = np.zeros(erode.shape, dtype=np.uint8)
    houghLines = cv2.HoughLinesP(erode, rho=20, theta=np.pi/180*7, threshold=1, minLineLength=100, maxLineGap=40)
    if type(houghLines) == type(None):
        return [], True, 0
    
    lines = np.zeros((len(houghLines),4))
    idx=0
    for line in houghLines:
        x1,y1,x2,y2 = line[0]
        lines[idx] = [x1,y1,x2,y2]
        idx+=1

    lines = merge(lines, MAX_MERGE_DIST, MAX_MERGE_ANGLE)
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(raw_hough_img, (int(x1), int(y1)),(int(x2), int(y2)) , [255, 0, 0],1)
    if show:
        cv2.imshow("merged lines", raw_hough_img)
    lanes, bot_dist, heading = pair_lines(
        lines, image, [MIN_TOP, MAX_TOP, MIN_BOT, MAX_BOT],
        [prev_heading, DISPLACEMENT_TOL, HEADING_DISP, prev_left, prev_right, ANG_TOL]
    )
    for lane in lanes:
        x1, y1, x2, y2 = lane[:4]
        cv2.line(image, (int(x1), int(y1)),(int(x2), int(y2)) , [0, 255, 200],3)
    
    if show:
        cv2.imshow('Lanes: ', image)
        cv2.waitKey()
    prev_left, prev_right = [], []
    for lane in lanes:
        # print(lane)
        if lane[5] == 0: 
            prev_left = lane
        else: prev_right = lane
    # cv2.imwrite(f"frames/{frame_num[0]}.jpg", image)
    # frame_num[0] = frame_num[0]+1
    return lanes, False, heading

if __name__ == "__main__":
    args = sys.argv[1:]
    # get_lanes(cv2.imread(args[0]), args[0], args[1])
    junction(cv2.imread(args[0]))

# if __name__ == "__main__":
#     path = "data_2023/AI_data"
#     img_names = os.listdir(path)
#     for name in img_names:
#         img = cv2.imread(f"{path}/{name}")
#         get_lanes(img, name)