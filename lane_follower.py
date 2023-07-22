# Give a summary of the code.

import cv2
import numpy as np
import time
import math
from picamera2 import Picamera2
import PID

# Initialise PID
P = 0.5
I = 0
D = 0

pid = PID.PID(P, I, D)
pid.SetPoint = 0
pid.setSampleTime(1)

def lane_det(prev_h):
    frame = cv2.imread("frame.jpg", cv2.IMREAD_GRAYSCALE)
    print("res", frame.shape)
    # frame = cv2.rotate(frame, cv2.ROTATE_180)
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower_blue = np.array([0, 100, 40])
    # upper_blue = np.array([175, 300, 175])
    # mask = cv2.inRange(hsv, lower_blue, upper_blue) # filter out wanted colours only (lanes)
    cropped_edges, line_segments = line_segs(frame)
    if type(line_segments) != type(np.array([])):
        print("Line_segs: None")
        return prev_h

    line_image = frame
    for line in line_segments:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (125, 0, 0), 5)
    if __name__ == "__main__":
        cv2.imshow("line_image", line_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows
    
    line_segments = line_segments[:4]
    lane_lines, lane_image = get_lanes(line_segments, frame)
    if type(lane_lines) != type([]):
        print("Lane_lines: None")
        return prev_h
    if __name__ == "__main__":
        cv2.imshow("laneimg", lane_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows

    angle = heading(lane_lines, frame, lane_image, prev_h)

    # if abs(angle) > 20:
    #     angle = 20 * angle/abs(angle)

    return angle

def line_segs(frame):    # Use mask to find and return line segments
    edges = cv2.Canny(frame, 190, 400, L2gradient =True)
    cropped_edges= focus(edges, np.zeros_like(edges))
    """-----------------------------------------------------------------"""
    if __name__ == "__main__":
        cv2.imshow("frame", frame)
        cv2.imshow("edges", edges)
        cv2.imshow("cropped_edges", cropped_edges)
        cv2.waitKey(0)
    """
    -----------------------------------------------------------------
    Parameters for extracting line segments
    """
    rho = 10    # distance precision in pixel, i.e. 1 pixel
    angle = (np.pi/180) * 4   # angular precision in radian, i.e. 1 degree
    min_threshold = 2  # Minimum number of votes
    minLineLength = 150
    maxLineGap = 60
    """---------------------------------------------------------------------"""
    line_segments = cv2.HoughLinesP(image=cropped_edges, rho=rho, theta=angle, threshold=min_threshold, 
        minLineLength=minLineLength, maxLineGap=maxLineGap)
    # print("line_segments:", line_segments)
    print("line segments: ", line_segments)
    
    return cropped_edges, line_segments

# Feeding the data to the PID controler and returning a final stear angle for the robot
def heading(lane_lines, frame, lane_image, prev_h):
    height, width = frame.shape
    mid_x = 0
    mid_y = 0
    slope = 0

    if len(lane_lines) == 1: 
        _, _, _, _, slope = lane_lines[0][0]
        # print("width, height: ", frame.shape)
        # print("x1, y1, x2, y2: ", lane_lines[0][0])
        # slope = (x2-x1) / (y2-y1)
        # print("slope: ", slope)
        mid_y =  height - abs((width/2 * slope))
        if slope < 0:
            mid_x = width

        # print("mid_y: ", mid_y)
        rads = math.atan(slope)
        angle = rads*180/np.pi
    else:
        _, _, left_x2, l_y2, _ = lane_lines[0][0]
        _, _, right_x2, r_y2, _ = lane_lines[1][0]
        mid_x = (left_x2+right_x2)/2
        mid_y = (l_y2+r_y2)/2
        assert mid_y == l_y2
        if int(mid_x) == int(width/2):
            angle = 90
        else:
            slope =  (int(mid_y)-int(height)) / (int(mid_x) - int(width/2))
            # print(slope)
            rads = math.atan(slope)
            angle = rads*180/np.pi
    if angle < 0:
        angle = -angle - 90
    else:
        angle = 90 - angle

    # print("angle: ", angle)
    cv2.line(lane_image, (int(width/2), int(height)), (int(mid_x), int(mid_y)), (100, 255, 255), 10)
    pid.update(angle)
    # cv2.imshow("heading", lane_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("pid.output: " + str(pid.output), "\n")
    return pid.output
    # if abs(angle) < 10 or (prev_lane == 2 and len(lane_lines) == 1) :
    #     angle = prev_h + angle
    # else:
    #         if len(lane_lines) == 2:
    #             angle = prev_h + (10 * angle/abs(angle))
    #         else:
    #             angle = prev_h = (4 * angle/abs(angle))


def make_points(frame, line):
    height, width = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1/2)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2, slope]]

def focus(edges, mask):
    # only focus bottom half of the screen
    height, width = edges.shape
    polygon = np.array([[
        (0, height*1/3),
        (width, height*1/3),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    # cv2.  ("image", cropped_edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cropped_edges


def get_lanes(line_segments, frame):
    lane_lines = []
    height, width = frame.shape
    left_fit = []
    right_fit = []
    # boundary = 1/2
    # left_region_boundary = width * (1 - boundary)
    # right_region_boundary = width * boundary
    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    left_fit_average = np.mean(left_fit, axis=0)
    print("left_fit", left_fit)
    print("left_fit_average", left_fit_average)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.mean(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))
    if len(lane_lines) == 0:
        return None, None
    lane_image = frame
    for lane in lane_lines:
        for x1, y1, x2, y2, _ in lane:
            cv2.line(lane_image, (x1, y1), (x2, y2), (200, 150, 150), 2)
    print("lane_lines", lane_lines)
    return lane_lines, lane_image

def direct():
    picam2 = Picamera2()
    camera_config = picam2.create_still_configuration(main={"size": (1640, 1232)})
    picam2.configure(camera_config)
    picam2.start()
    time.sleep(2)
    picam2.capture_file("frame.jpg")
    picam2.close()
    print(lane_det(0))

if __name__ == "__main__":
    direct()