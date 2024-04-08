import cv2
import sys
import numpy as np
import math
from lane_func import *
from merge_lines import mergeLine
import os
from copy import deepcopy
import time

MAX_MERGE_DIST, MAX_MERGE_ANGLE = 100, 0.3
DIST_TOL = 1
MIN_JUNC_DIST = 100
MAX_JUNC_DIST = 1000
scale = 1

def junction(img, Ev3, show=(__name__=="__main__")):
    line_img = deepcopy(img)
#     img = delete_top(line_img, crop_fraction=1/7)
    width, height, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    if True:
        cv2.imshow("thresh", thresh)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    houghLines = cv2.HoughLinesP(thresh, rho=1, theta=np.pi/180*0.7, threshold=30, minLineLength=50, maxLineGap=10)
    if type(houghLines) == type(None):
        return
    print(len(houghLines))
    lines = np.zeros((len(houghLines),4))
    for idx, line in enumerate(houghLines):
        x1,y1,x2,y2 = line[0]
        lines[idx] = [x1,y1,x2,y2]

    lines = mergeLine(lines, MAX_MERGE_DIST, MAX_MERGE_ANGLE)
    lines = list(lines)
    if True:
        for line in lines:
            cv2.line(img, [int(x) for x in line[:2]], [int(x) for x in line[2:]], (245, 0, 200), 3)
            cv2.imshow("img", img)
            cv2.waitKey()

    if len(lines) < 5:
        return
    intersects = get_intersects(lines, 100)
    intersects = sorted(intersects, key=lambda x:(x[0][0], x[0][1]))
    print("================================\n", intersects)
    if len(intersects) == 2:
        if intersects[0][0][1] > height/2 and intersects[1][0][1] > height/2:
            left_point, right_point = intersects
            # T_junction: turn left or right.
            pass
        
        top_point, bottom_point = [], []
        if intersects[0][0][1] < intersects[1][0][1]:
            top_point, bottom_point = intersects
        else:
            top_point, bottom_point = intersects[1], intersects[0]

        if intersects[1][0][0] < width/2: # because it is sorted, we don't need to test the other point.
            print("Left T junction: turn left or straight")
            line3 = max(lines[bottom_point[1]], lines[bottom_point[2]], key= lambda x : (x[1]+x[3]))
            idxs = list(set([bottom_point[1], bottom_point[2], top_point[1], top_point[2]]))
            idxs = sorted(idxs, reverse=True)
            for idx in idxs:
                lines.pop(idx)
            line4 = lines[0] # all other lines have been removed
            if True:
                for point in intersects:
                    cv2.circle(img, [int(x) for x in point[0]], 5, (255, 0, 0), -1)
                cv2.line(img, [int(x) for x in line4[:2]], [int(x) for x in line4[2:]], (0, 0, 0), 5)
                cv2.imshow("line4", img)
                cv2.line(img, [int(x) for x in line3[:2]], [int(x) for x in line3[2:]], (0, 0, 0), 5)
                cv2.imshow("line3", img)
            left(line3, line4, bottom_point, img.shape, Ev3)
        else:
            # Right T junction: turn right or straight
            pass
    elif len(intersects) == 4:
        #crossroad: left, right or straight
        pass
    return


"""
__| |__
__  |__
  | |
 
 line3: the top left horizontal line
 line4: the long veritcal line on the right. (3 segments)
 Point1: the bottom left line
 img_shape = [height, width] of the image
 Ev3: stdin of Paramiko agent of ev3 connection.
 
 turn consits of sequence of 4 things.
 turn1 => rotate to properly line up with track.
 move1 => Move forward until at the location to turn at.
 turn2 => take the turn itself. Now facing in-betweeen line 2 and 3
 move2 => move forward a little before letting PID take-over again.
"""
    
def left(line3, line4, point1, img_shape, Ev3):
    turn1 = get_line_ang(line4)
#     Ev3.write(f"t\n{int(turn1)}\n")
#     time.sleep(0.5)
    move1_pxl = img_shape[0] - point1[1] #Number of pixels for move 1.
    move1 = move1_pxl * scale #scale is the ratio of 1 pixel : mm
#     Ev3.write(f"m\n{move1}\n")
#     time.sleep(move1/60 +1)
    s3, m3 = get_slope(line3)
    s4, m4 = get_slope(line4)
    turn2 = math.degrees(math.atan((m4-m3)/(1+(m3*m4))))
    print("___\n | \n | \n ___")
    print(turn1, turn2, move1)
    Ev3.write("s\n")
    cv2.waitKey()
    exit()
#     Ev3.write(f"t\n{turn2}\n")
#     time.sleep(0.5)
#     Ev3.write("m \n 60 \n ")
#     time.sleep(1.5)