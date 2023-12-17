import cv2
import math
import numpy as np

# LEN_WEIGHT = 1.3
# WIDTH_WEIGHT = 1
MIN_LANE_LENGTH = 500
MAX_LANE_WIDTH = 100
VOTE_WIDTH = 50
MIN_P_TO_A = 0.05
LANE_COLOR = (200,100,100)
CURVE_AREA_THRESH = 0.4
MIN_LANE_GAP = {"max":750, "min": 350}
# MAX_LANE_GAP = {"max":200, "min": 100}

def get_votes(image, cnt, idx, testing):
    votes = 1

    # cnt_len = cv2.arcLength(cnt, False)
    rect = cv2.minAreaRect(cnt)
    height = rect[1][0]
    width = rect[1][1]
    if width > height:
        width, height = height, width
    perimeter = 2*(width+height)
    box_area = width*height
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(image,[box],0,(0,0,255),2)

    if height < MIN_LANE_LENGTH: votes = 0
    elif width > MAX_LANE_WIDTH:
        cnt_area = cv2.contourArea(cnt)
        ratio = cnt_area/box_area
        print("ratio: ", ratio)
        if ratio > CURVE_AREA_THRESH:
            votes = 0
    elif width < VOTE_WIDTH: votes += 1
    # if perimeter/box_area > MIN_P_TO_A: votes += 1

    print("index, votes:  ", idx, votes)
    print("len, width:  ", height, width)
    """
    votes = LEN_WEIGHT*cnt_len+WIDTH_WEIGHT/cnt_width
    slope, intercept, angle = fit_and_angle(cnt, image, testing)
    """
    return votes, height

def fit_and_angle(cnt, image, testing):
    [vx,vy,x,y] = cv2.fitLine(cnt,cv2.DIST_L2,0,0.01,0.01)
    print(vx,vy,x,y)
    lefty = int((-x*vy/vx) + y)
    righty = int(((image.shape[1]-x)*vy/vx)+y)
    if testing:
        cv2.line(image,(image.shape[1]-1,righty),(0,lefty),LANE_COLOR,2)
        cv2.imshow("fitted line", image)
        cv2.waitKey()

    slope = (lefty-righty)/(0-(image.shape[1]-1))
    angle = math.atan(slope) * 180/np.pi
    # translate angles from topleft to bottom left refernce
    if (angle < 0):
        angle = -1 * angle
    else:
        angle = 180 - angle
    print("\n slope,angle", slope, angle,"\n")

    """
    c = y - mx
    intercept = lefty + 0*slope
    """
    lane_points = np.array(((image.shape[1]-1,righty),(0,lefty)), dtype=np.int32 )
    return slope, lefty, angle, lane_points

def cnt_is_blocking(leftmost, rightmost, slope, intercept, angle):
    if angle == 90:
        lane_x1 = (leftmost[1]-intercept)/slope
        lane_x2 = (rightmost[1]-intercept)/slope
        if lane_x1>leftmost[1] and lane_x2<rightmost[1]:
            return True
        return False
    elif angle < 90:
        cnt_x, cnt_y = rightmost
        lane_x = (cnt_y-intercept)/slope
        if cnt_x > lane_x:
            return True
    else:
        cnt_x, cnt_y = leftmost
        lane_x = (cnt_y-intercept)/slope
        if cnt_x < lane_x:
            return True
    
    return False

# checks if two given lines can be a lane based on distance
def check_pair(line1, line2, im_width, im_height):
    cnt1 = line1[0]
    cnt2 = line2[0]
    # print("\n\n", line1[0])
    # print("\n\nContour: \n", cnt1, "\n___________________________________\n")

    s1, i1 = line1[3], line1[4]
    s2, i2 = line2[3], line2[4]
    print("s1, s2, i1, i2", s1, s2, i1, i2)
    if s1 != s2:
        com_x = (i2 - i1)/(s1 - s2)
        com_y = com_x * s1 + i1
        if 0 < com_x < im_width and 0 < com_y < im_height:
            return 0

    line1 = [[(im_height-i1)/s1, im_height], [(-i1)/s1, 0]]
    line2 = [[(im_height-i2)/s2, im_height], [(-i2)/s2, 0]]

    min_distance = np.inf
    for point1 in line1:
        for point2 in line2:
            min_distance = min(min_distance, abs(int(point2[0]-point1[0])))
    
    print("\n MIN DISTANCE: ", min_distance, "\n")
    if  MIN_LANE_GAP["min"] < min_distance < MIN_LANE_GAP["max"]:
        return 1

    return 0