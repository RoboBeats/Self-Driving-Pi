import cv2
import sys
import numpy as np
import math
from lane_func import *
from merge_lines import mergeLine
import os
from copy import deepcopy

MAX_MERGE_DIST, MAX_MERGE_ANGLE = 100, 0.5
# for a piar of lanes
MIN_TOP, MAX_TOP = 200, 500
MIN_BOT, MAX_BOT = 400, 600
# for single lane:
ANG_TOL = 20
prev_right = []
prev_left = []
DISPLACEMENT_TOL = 50
HEADING_DISP = 5

MIN_LANE_LENGTH = (200, 0, 100)
MAX_LANE_WIDTH = 50        #Max lane marker width, if more than that some other object
LANE_COLOR = (200,100,100)
NON_LANE_COLOR = (200, 0,150)
MIN_BLOCK_AREA = 75000

# For junctions:
DIST_TOL = 1
MIN_JUNC_DIST = 100
MAX_JUNC_DIST = 1000

frame_num = [0]

def get_lanes(original_image, img_name, prev_heading, prev_left=prev_left, prev_right=prev_right, show=(__name__=="__main__")):
    # Load image, delete top portion, grayscale, Otsu's threshold
    image = delete_top(original_image)
    print(image.shape)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if show:
        cv2.imshow('original',original_image)
        cv2.imshow('gray_resized', gray)
        cv2.waitKey()
#         cv2.destroyAllWindows()
    
    #ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    if show:
        cv2.imshow('thresh', thresh)
        cv2.waitKey()

    raw_hough_img = np.zeros(thresh.shape, dtype=np.uint8)
    houghLines = cv2.HoughLinesP(thresh, rho=20, theta=np.pi/180*7, threshold=1, minLineLength=100, maxLineGap=40)
    print(houghLines)
    if type(houghLines) == type(None):
        # cv2.imwrite(f'hough_lines_test/on_close/{img_name}', raw_hough_img)
        return [], True, 0
    
    # print("blank shape: ",raw_hough_img.shape)
    # print("Number of houghLines:", len(houghLines))
    lines = np.zeros((len(houghLines),4))
    idx=0
    for line in houghLines:
        x1,y1,x2,y2 = line[0]
        # cv2.line(raw_hough_img, (x1, y1),(x2, y2) , [255, 0, 0],1)
        lines[idx] = [x1,y1,x2,y2]
        idx+=1
    # print(lines)
    # cv2.imwrite(f'hough_lines_test/on_close/{img_name}', raw_hough_img)

    lines = mergeLine(lines, MAX_MERGE_DIST, MAX_MERGE_ANGLE)
    print("----merged lines---------k")
    print(lines)
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(raw_hough_img, (int(x1), int(y1)),(int(x2), int(y2)) , [255, 0, 0],1)
    if show:
        cv2.imshow(f'line idx: {np.where(lines==line)[0][0]}', raw_hough_img)
        cv2.waitKey()
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