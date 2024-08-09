import cv2
import sys
import numpy as np
import math
from lane_func import *
from merge_linesv2 import merge
import os
from copy import deepcopy
import time

MAX_MERGE_DIST, MAX_MERGE_ANGLE = 50, 20
DIST_TOL = 25
MIN_JUNC_DIST = 100
MAX_JUNC_DIST = 1000
ang_scale = 1

def scale(x):
    x *= 0.78
    return x

direction = input()
if direction == "school":
    turns = ['w', 'a', 'w']
elif direction == "cafe":
    turns == ['s', 'd']
elif direction == 'lecture':
    turns = ['a', 'd']
elif direction == "audi":
    turns = ['d', 's']

def get_intersects(lines, dist_tol):
    intersects = []
    for i, line1 in enumerate(lines[:-1]):
        for j, line2 in enumerate(lines[i+1:]):
            dist1 = math.dist(line1[:2], line2[:2])
            if dist1 < dist_tol:
                x = (line1[0]+line2[0])/2
                y = (line1[1]+line2[1])/2
                intersects.append([[x, y], i, i+j+1])
                break
            
            dist2 = math.dist(line1[:2], line2[2:])
            if dist2 < dist_tol:
                x = (line1[0]+line2[2])/2
                y = (line1[1]+line2[3])/2
                intersects.append([[x, y], i, i+j+1])
                break
            
            dist3 = math.dist(line1[2:], line2[:2])
            if dist3 < dist_tol:
                x = (line1[2]+line2[0])/2
                y = (line1[3]+line2[1])/2
                intersects.append([[x, y], i, i+j+1])
                break
            
            dist4 = math.dist(line1[2:], line2[2:])
            if dist4 < dist_tol:
                x = (line1[2]+line2[2])/2
                y = (line1[3]+line2[3])/2
                intersects.append([[x, y], i, i+j+1])
#             print(dist1, dist2, dist3, dist4, 'distances _____________________-')
                
    return intersects

def transform(img, height, width, warp_frac = 1/4):
    src = np.array( [[0, 0], [width-1, 0], [0, height-1], [width-1, height-1]]).astype(np.float32)
    dst = np.array( [[0, 0], [width-1, 0], [width*warp_frac, height-1], [width*(1-warp_frac), height-1]] ).astype(np.float32)
    white_image = np.full((height, width, 3), 255, dtype=np.uint8)
    transform_mat = cv2.getPerspectiveTransform(src, dst)
    img = cv2.warpPerspective(img, transform_mat, (width, height), white_image, borderMode=cv2.BORDER_TRANSPARENT)
    return img


def junction(img, Ev3, show=(__name__=="__main__")):
    print("img shape: ", img.shape)
    line_img = deepcopy(img)
    height, width, _ = img.shape
    warp = transform(img, height, width)
    warp = delete_top(warp, crop_fraction=1/6)
    mask = cv2.inRange(warp, np.array([2, -1, -1]), np.array([130, 130, 130]))
    kernel = np.ones((7, 7), np.uint8)
    erode = cv2.erode(mask, kernel)

    if show:
        cv2.imshow("img", mask)
        cv2.imshow("thresh_post-cropping", erode)
        cv2.waitKey()
        cv2.destroyAllWindows()
    houghLines = cv2.HoughLinesP(erode, rho=1, theta=np.pi/180*0.7, threshold=40, minLineLength=40, maxLineGap=10)
    if type(houghLines) == type(None):
        print("returned, no juncntion linens")
        return
    print(len(houghLines))
    lines = np.zeros((len(houghLines),4))
    for idx, line in enumerate(houghLines):
        x1,y1,x2,y2 = line[0]
        lines[idx] = [x1,y1,x2,y2]
        if show:
            cv2.line(warp, [int(x) for x in lines[idx][:2]], [int(x) for x in lines[idx][2:]], (0, 0, 200), 10)
    if show:
        cv2.imshow("img", warp)
        cv2.waitKey()

    lines = merge(lines, MAX_MERGE_DIST, MAX_MERGE_ANGLE, im_shape=warp)
    lines = list(lines)
    if show:
        for line in lines:
            cv2.line(warp, [int(x) for x in line[:2]], [int(x) for x in line[2:]], (0, 200, 0), 2)
            print(line)
            cv2.imshow("img", warp)
            cv2.waitKey()
        cv2.imshow("img", warp)
        cv2.waitKey()

    if len(lines) < 5:
        return
    intersects = get_intersects(lines, DIST_TOL)
    intersects = sorted(intersects, key=lambda x:(x[0][0], x[0][1]))
    print("================================\n", intersects)
    if show:
        for intersect in intersects:
            print(intersect)
            warp = cv2.circle(warp, list(map(int, intersect[0])), 5, (200, 0, 0), -1)
        cv2.imshow("intersects", warp)
        print("hello")
        cv2.waitKey()
    for point in intersects:
        ang = two_line_ang(lines[point[1]], lines[point[2]])
        print("intersect angle is: ", ang)
        if ang < 60:
            return
        elif ang > 110:
            return
    select_path(Ev3, intersects, img.shape, lines)

def which_direction(turn, direction=direction):
    if direction:
        if len(turns) == 0:
            exit()
        turn = turns[0]
        turns.pop(0)
        return turn
    print(f"There is a {turn} ahead. Use WAD keys to select path")
    print("t-turn: turn left or right")
    print("left T turn: turn left or straight")
    print("right T turn: turn right or straight")
    print("Crossroad: turn left, right, or straight.")
    direction = input()
    return direction

# uses intersection points to determine the type of junction, and then move left, right, or straight.
def select_path(Ev3, intersects, img_shape, lines):
    print(lines)
    height, width, _ = img_shape
    intersects = sorted(intersects, key = lambda x:x[0][1])
    if len(intersects) == 2:
        if Ev3 !=  None:
            Ev3.write("s \n")
        if len(lines) > 5:
            return
        if intersects[0][0][1] > height/2 and intersects[1][0][1] > height/2:
            l_point, r_point = intersects
            # T_junction: turn left or right.
            if which_direction("T_junction") in ["A", 'a']: # turn left
                line1, line2 = sorted([lines[l_point[1]], lines[l_point[2]]],
                                      key = lambda x:(x[1]+x[3]), reverse=True)
                line3 = max([lines[r_point[1]], lines[r_point[2]]],
                                      key = lambda x:(x[1]+x[3]))
                left(line1, line2, line3, l_point, img_shape, Ev3)
            else: # turn right
                line1, line2 = sorted([lines[r_point[1]], lines[r_point[2]]],
                                      key = lambda x:(x[1]+x[3]), reverse=True)
                line3 = max([lines[l_point[1]], lines[l_point[2]]],
                            key = lambda x:(x[1]+x[3]))
                right(line1, line2, line3, r_point, img_shape, Ev3)
            return
        if abs(intersects[0][0][1] - intersects[1][0][1]) < 100:
            return
        if intersects[0][0][1] < intersects[1][0][1]:
            top_point, bot_point = intersects
        else:
            top_point, bot_point = intersects[1], intersects[0]

        if intersects[1][0][0] < width/2:
            line1, line2, line3, line4 = sorted([lines[bot_point[1]], lines[bot_point[2]], lines[top_point[1]], lines[top_point[2]]],
                                      key = lambda x:(x[1]+x[3]), reverse=True)
            
            if which_direction("Left T-Turn") in ["A", 'a']:
                line1, line2 = sorted([lines[bot_point[1]], lines[bot_point[2]]],
                                      key = lambda x:(x[1]+x[3]), reverse=True)
                idx = [x for x in range(5)]
                for i in bot_point[1:]+top_point[1:]:
                    idx.remove(i)
                left(line1, line2, lines[idx[0]], bot_point, img_shape, Ev3)
            else:
                straight(lines, img_shape, Ev3)
        else:
            # Right T junction: turn right or straight
            if which_direction("Right T-Turn") in ["D", 'd']:
                line1, line2 = sorted([lines[bot_point[1]], lines[bot_point[2]]],
                                      key = lambda x:(x[1]+x[3]), reverse=True)
                print("right, ", line1, line2)
                idx = [x for x in range(5)]
                for i in bot_point[1:]+top_point[1:]:
                    print(i)
                    idx.remove(i)
                right(line1, line2, lines[idx[0]], bot_point, img_shape, Ev3)
            else:
                straight(lines, img_shape, Ev3)
    elif len(intersects) == 4:
        #crossroad: left, right or straight
        if Ev3 != None:
            Ev3.write("s\n")
        t_points = intersects[:2]
        b_points = intersects[2:]
        b_l_point, b_r_point = sorted(b_points, key = lambda x:x[0][0])
        t_lpoint, t_r_point = sorted(t_points, key = lambda x:x[0][0])
        turn = which_direction("Crossroad")
        if turn in ["W", 'w']:
            straight(lines, img_shape, Ev3)
        elif turn in ["A", 'a']:
            line1, line2 = sorted([lines[b_l_point[1]], lines[b_l_point[2]]],
                                  key = lambda x:(x[1]+x[3]), reverse=True)
            line3 = max([lines[b_r_point[1]], lines[b_r_point[2]]],
                                  key = lambda x:(x[1]+x[3]))
            left(line1, line2, line3, b_l_point, img_shape, Ev3)
        else:
            line1, line2 = sorted([lines[b_r_point[1]], lines[b_r_point[2]]],
                                  key = lambda x:(x[1]+x[3]), reverse=True)
            line3 = max([lines[b_l_point[1]], lines[b_l_point[2]]],
                                  key = lambda x:(x[1]+x[3]))
            right(line1, line2, line3, b_r_point, img_shape, Ev3)

"""
    __| |
    __  |
      | |
 
 * The junction might be different, this is for representation purpose only.
 
 line1: the bottom left vertical line
 line2: the bottom left horizontal lane
 Point1: the bottom left point
 img_shape = [height, width] of the image
 Ev3: stdin of Paramiko agent of ev3 connection.
 
 turn consits of sequence of 4 things.
 turn1 => rotate to properly line up with track.
 move1 => Move forward until at the location to turn at.
 turn2 => take the turn itself. Now facing in-betweeen line 2 and 3
 move2 => move forward a little before letting PID take-over again.
         Ensures lanes are clearly in view and no interference
"""

def left(line1, line2, line3, point1, img_shape, Ev3):
    print(line1, line2, point1)
    ang_l1 = get_line_ang(line1)
    ang_l3 = get_line_ang(line3)
    ang_l2 = get_line_ang(line2)
    move1_pxl = img_shape[0] - point1[1] #Number of pixels for move 1.
    move1 = scale(move1_pxl) #scale is the ratio of 1 pixel.
    turn1 = (ang_l1+ang_l3)/2	
    turn2 = ang_l2*ang_scale
    if turn2 > 0: turn2 -= 180
    turn2 -= turn1
    print("Turn Values: ", turn1, turn2, move1)
    Ev3.write(f"t\n{int(turn1)}\n")
    time.sleep(0.05)
    Ev3.write(f"m\n{int(move1)}\n")
    time.sleep(move1/70)
    Ev3.write(f"t\n{int(turn2)}\n")
    time.sleep(0.05)
    Ev3.write("m \n 60 \n ")
    time.sleep(0.5)

"""
    | |__
    |  __
    | |
 
 * The junction might be different, this is for representation purpose only.
 
 line1: the bottom right vertical line
 line2: the bottom right horizontal line
 Point1: the bottom right point
 img_shape = [height, width] of the image
 Ev3: stdin of Paramiko agent of ev3 connection.
 
 turn consits of sequence of 4 things.
 turn1 => rotate to properly line up with track.
 move1 => Move forward until at the location to turn at.
 turn2 => take the turn itself. Now facing in-betweeen line 2 and 3
 move2 => move forward a little before letting PID take-over again.
 Ensures lanes are clearly in view and no interference
"""

def right(line1, line2, line3, point1, img_shape, Ev3):
    print("\n_____________________________\n")
    print(line1, line2)
    ang_l1 = get_line_ang(line1)
    ang_l2 = get_line_ang(line2)
    ang_l3 = get_line_ang(line3)
    move1_pxl = img_shape[0] - point1[1] #Number of pixels for move 1.
    move1 = scale(move1_pxl) #scale is the ratio 1 mm:1 pixel.
    turn1 = (ang_l1 +ang_l3)/2
    turn2 = ang_l2*ang_scale
    if turn2 < 0: turn2 += 180
    print("Turn Values: ", turn1, turn2, move1)
    Ev3.write(f"t\n{int(turn1)}\n")
    time.sleep(0.05)
    Ev3.write(f"m\n{int(move1)}\n")
    time.sleep(move1/60)
    print("a")
    Ev3.write(f"t\n{int(turn2)}\n")
    time.sleep(0.5)
    print("b")
    Ev3.write("m \n 60 \n")
    time.sleep(1)
    print("c")

"""takes the two bottom-most lines, aligns itself, and goes."""
def straight(lines, img_shape, Ev3):
    print("straight")
    a = sorted(lines, key = lambda x:max(x[1], x[3]), reverse = True)
    line1, line2 = a[:2]
    ang_l1 = get_line_ang(line1)
    ang_l2 = get_line_ang(line2)
    turn1 = (ang_l1+ang_l2)/2
    turn1 *= 1.1
    time.sleep(0.05)
    move_1 = img_shape[0]-(line1[1]+line2[1])/2 + 400
    move_1 = scale(move_1)
    Ev3.write(f"t\n{int(turn1)}\n")
    Ev3.write(f"m\n{int(move_1)}\n")
    time.sleep(move_1/70)
