import cv2
import numpy as np
import math

"""
0 means left and 1 means right for lanes.
Shape of a lane: [x1, y1, x2, y2, x_bot, lane side, angle]

accross everything, img.shape = [height, width, channels]
"""

def delete_top(image, crop_fraction = 3/5):
    shape = image.shape
    height = shape[0]
    if len(shape) == 2:
        resized_image = np.delete(image, slice(int(height*crop_fraction)),0)
    else:
        resized_image = np.delete(image, slice(int(height*crop_fraction)),0)
#     print('input image dimension:', image.shape, '  resized image dimensions:', resized_image.shape)
    return resized_image

def get_slope(line):
    x1, y1, x2, y2 = line[:4]
    slope = np.inf if x2==x1 else (y2-y1)/(x2-x1)
    intercept = y1 - slope*x1
    return slope, intercept

def get_line_ang(line):
    slope, _ = get_slope(line)
    angle = math.atan(slope) * 180/np.pi
    if (angle < 0):
        angle *= -1
    else:
        angle = 180 - angle
    angle = 90 - angle
    return -angle

def two_line_ang(line1, line2):
    a1, a2 = get_line_ang(line1), get_line_ang(line2)
    ang = abs(a1-a2)
    return ang

def get_top_and_bottom(line, img):
    height, width, _ = img.shape
    slope, intercept = get_slope(line[:4])
    if slope == 0:
        return line[0], line[0]
    x_top = (0-intercept)/slope
    x_bot = (height-intercept)/slope
    return x_top, x_bot

def pair_lines(lines, img, pair_params, single_lane_params): #takes lines(2+), and finds the best pair to make into lanes.
    min_top, max_top, min_bot, max_bot = pair_params
    if len(lines) < 2:
        return single_lane(lines, single_lane_params, img)
    pairs = []
    for idx, line1 in enumerate(lines[:-1]):
        x_top_1, x_bot_1 = get_top_and_bottom(line1, img)
        for line2 in lines[idx+1:]:
            x_top_2, x_bot_2 = get_top_and_bottom(line2, img)
            top_dist = abs(x_top_1 - x_top_2)
            bot_dist = abs(x_bot_1 - x_bot_2)
            votes = 0

            if min_top < top_dist < max_top:
                votes += 1
            if min_bot < bot_dist < max_bot:
                votes += 1
            if votes ==2:
                line1 = list(line1)
                line2 = list(line2)
                line1.append(x_bot_1)
                line2.append(x_bot_2)
                if x_bot_1 < x_bot_2:
                    line1.append(0)
                    line2.append(1)
                else:
                    line1.append(1)
                    line2.append(0)
                pairs.append([
                    line1, line2, votes, top_dist, bot_dist, x_bot_1, x_bot_2
                ])
    if len(pairs) == 0:
        return single_lane(lines, single_lane_params, img)
    pair = pairs[0]
    heading = ang_from_pair(pair[0], pair[1], img.shape)
    return pair[:2], pair[-2:], heading

def ang_from_pair(line1, line2, img_shape):
    # get slope and intercept of both lines
    m1, b1 = get_slope(line1)
    m2, b2 = get_slope(line2)
    #get point (com_x, com_y) where lines meet
    if m1-m2 == 0:
        return 0
    com_x = (b2-b1)/(m1-m2)
    com_y = com_x*m1 + b1
    return get_line_ang([img_shape[1]/2, img_shape[0], com_x, com_y])

def single_lane(lines, single_lane_params, img):
    if type(lines) == type(np.array([])):
        lines = lines.tolist()
    prev_heading, x_tol, heading_disp, prev_left, prev_right, ang_tol = single_lane_params
    ideal_lanes = []
    for line in lines:
        angle = get_line_ang(line)
        _, x_bot = get_top_and_bottom(line, img)
        line.append(x_bot)
        if x_bot < img.shape[0]/2:
            line.append(0)
        else: line.append(1)
        line.append(angle)
        if (line[5] == 0 and len(prev_left)) or line[5] == 1 and len(prev_right) == 0:
            continue
        prev_lane = []
        if line[6] == 0:
            prev_lane = prev_left
        else: prev_lane = prev_right
        if prev_lane == []:
            continue
        ideal_ang = prev_lane[5]+prev_heading
        if ideal_ang-ang_tol < angle < ideal_ang+ang_tol:
            ideal_x = prev_lane[4] + (heading_disp * prev_heading)
            if ideal_x-x_tol < x_bot < ideal_x+x_tol:
                return [line[0]], [line[-1]]
            ideal_lanes.append(line)
    if len(ideal_lanes) == 0:
        return [lines[0]], [lines[0][-1]], lines[0][-1]
    return [ideal_lanes[0]], [ideal_lanes[0][-1]], ideal_lanes[0][-1]