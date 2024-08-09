import cv2
import numpy as np
from lane_func import *
import math

# Return true if line segments AB and CD intersect
def orientation(p, q, r):
    """Return the orientation of the triplet (p, q, r).
    0 -> p, q and r are collinear
    1 -> Clockwise
    2 -> Counterclockwise
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0
    elif val > 0:
        return 1
    else:
        return 2

def on_segment(p, q, r):
    """Check if point q lies on line segment 'pr'."""
    return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
            min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

def intersect(p1, q1, p2, q2):
    """Return True if line segments 'p1q1' and 'p2q2' intersect."""
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special Cases
    # p1, q1 and p2 are collinear and p2 lies on segment p1q1
    if o1 == 0 and on_segment(p1, p2, q1):
        return True

    # p1, q1 and p2 are collinear and q2 lies on segment p1q1
    if o2 == 0 and on_segment(p1, q2, q1):
        return True

    # p2, q2 and p1 are collinear and p1 lies on segment p2q2
    if o3 == 0 and on_segment(p2, p1, q2):
        return True

    # p2, q2 and q1 are collinear and q1 lies on segment p2q2
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False

"""a is the slope of the line. c is intercept [x1, y1] is any point"""
def perpendicular(p1, p2, p3):
    line = p1+p2
    a, c = get_slope(line)
    b = -1
    
    x1, y1 = p3
    temp = (-1 * (a * x1 + b * y1 + c) //
                  (a * a + b * b)) 
    x = temp * a + x1
    y = temp * b + y1
    if (min(p1[0], p2[0]) <= x <= max(p1[0], p2[0]) and
        (min(p1[1], p2[1]) <= y <= max(p1[1], p2[1]))):
        return point_dist(p3, [x, y])
    else:
        return min(point_dist(p1, p3), point_dist(p2, p3))


def point_dist(p1, p2):
    return math.dist(p1, p2)

def line_line(p1, p2, p3, p4):
    if intersect(p1, p2, p3, p4):
        return 0
    dist = [perpendicular(p1, p2, p3), perpendicular(p1, p2, p4),
            perpendicular(p3, p4, p1), perpendicular(p3, p4, p2)]
    dist = sorted(dist)
    return dist[0]

def merge(lines, max_dist, max_angle, im_shape=None):
    lines = lines.tolist()
    merged = 1
    current = 0
    while merged>0:
        merged = 0
        i, j = 0, 1
        while i < len(lines)-1:
            j = i+1
            while j < len(lines):
                line1 = lines[i]
                line2 =  lines[j]
                ang = two_line_ang(line1, line2)
                ang = min(ang, 180-ang)
                if  ang < max_angle:
                    p1, p2 = line1[:2], line1[2:]
                    p3, p4 = line2[:2], line2[2:]
                    dist = line_line(p1, p2, p3, p4)
                    if dist < max_dist:
                        merged += 1
                        d = [point_dist(p1, p2), point_dist(p1, p3), point_dist(p1, p4), point_dist(p2, p3),
                             point_dist(p2, p4), point_dist(p3, p4)]
                        if max(d) == d[0]:
                            line = p1 + p2
                        elif max(d) == d[1]:
                            line = p1 + p3
                        elif max(d) == d[2]:
                            line = p1 + p4
                        elif max(d) == d[3]:
                            line = p2 + p3
                        elif max(d) == d[4]:
                            line = p2 + p4
                        else:
                            line = p3 + p4
                        lines.pop(j)
                        lines[i] = line
                        if type(im_shape) != type(None) and current == 0:
                            img = np.zeros_like(im_shape)
                            cv2.line(img, [int(x) for x in line1[:2]],
                                     [int(x) for x in line1[2:]], (255, 0, 0), 3)
                            cv2.line(img, [int(x) for x in line2[:2]],
                                     [int(x) for x in line2[2:]], (255, 0, 0), 3)
                            cv2.line(img, [int(x) for x in line[:2]],
                                     [int(x) for x in line[2:]], (0, 0, 255), 1)
                            print("the lines being  displayed are: ", line1, line2, line)
                            cv2.imshow("img", img)
                            cv2.waitKey()
                            cv2.destroyAllWindows
                        continue
                j += 1
            i  += 1
        current=0

    return lines