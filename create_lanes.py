import cv2
import sys
import numpy as np
import math
from lane_func import *
from merge_lines import mergeLine
import os

MAX_MERGE_DIST, MAX_MERGE_ANGLE = 80, 10
# for a piar of lanes
MIN_TOP, MAX_TOP = 500, 1300
MIN_BOT, MAX_BOT = 900, 1850
# for single lane:
ANG_TOL = 20
prev_right = []
prev_left = []
DISPLACEMENT_TOL = 50
HEADING_DISP = 5

MIN_LANE_LENGTH = 500
MAX_LANE_WIDTH = 50        #Max lane marker width, if more than that some other object
LANE_COLOR = (200,100,100)
NON_LANE_COLOR = (100,100,100)
MIN_BLOCK_AREA = 75000

frame_num = [0]

def get_lanes(original_image, img_name, prev_heading, prev_left=prev_left, prev_right=prev_right, show=(__name__=="__main__")):
    # Load image, delete top portion, grayscale, Otsu's threshold
    image = delete_top(original_image)
    print(image.shape)

    #sharpening_kernel =np.array( [[-1, -1, -1],[-1,9,-1], [-1,-1,-1]])
    #sharpened = cv2.filter2D(image, -1, sharpening_kernel)
    #gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if show:
        cv2.imshow('original',original_image)
        cv2.imshow('gray_resized', gray)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    #ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    if show:
        cv2.imshow('thresh', thresh)
        cv2.waitKey()

    raw_hough_img = np.zeros(thresh.shape, dtype=np.uint8)
    houghLines = cv2.HoughLinesP(thresh, rho=20, theta=.15, threshold=127, minLineLength=250, maxLineGap=10)
    # print(houghLines)
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
    # print("----merged lines---------k")
    # print(lines)
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


def get_single_lane_old(image, cnt,idx):
    print("Entering contour id", idx)
    cnt_length = cv2.arcLength(cnt, False)
    cnt_area = cv2.contourArea(cnt)
    print("contour ", idx, " length ", cnt_length, " area: ", cnt_area)
    if cnt_length < MIN_LANE_LENGTH:
         print("ignoring contour ", idx, " length ", cnt_length, " below min lane length")
         return None, None, None
    
    # Create line mask and convex hull mask
    line_mask = np.zeros(image.shape, dtype=np.uint8)
    convex_mask = np.zeros(image.shape, dtype=np.uint8)
    contours_mask = np.zeros(image.shape, dtype=np.uint8)

    for point in cnt:
        for subpoint in point:
            contours_mask[subpoint[1]][subpoint[0]]=255
    
    if __name__ =='__main__':
        cv2.imshow(f'contours_mask_%d'%(idx), contours_mask)
        cv2.waitKey()

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    print("minAreaRect=> ",rect," <=")

    #check if this is a lane marker by checking width isgreate than a fraction of height
    width = min(rect[1][1], rect[1][0])
    height = max(rect[1][1], rect[1][0])
    MAX_AREA_THRESH = 3
    #if width > MAX_LANE_WIDTH:
    #if height < 700 and  (width * height > MAX_AREA_THRESH * cnt_area or width > MAX_LANE_WIDTH):
    if cnt_area > 13* cnt_length:
        #print(f'Ignoring contour %d, width %d more than max allowed %d' % (idx,width,MAX_LANE_WIDTH))
        print(f'Ignoring contour %d, min rect area dimensions %d %d, area %d,  more than  %d times cnt_area %d cnt_length %d' % (idx,width, height,width*height, MAX_AREA_THRESH, cnt_area, cnt_length))
        cv2.drawContours(contours_mask,[box],0,NON_LANE_COLOR,2)
        if __name__ =='__main__':
            cv2.imshow(f'contours_mask_with_boundingrect_%d'%(idx), contours_mask)
            cv2.waitKey()
        return None, None, None
    else:
        cv2.drawContours(contours_mask,[box],0,LANE_COLOR,2)
        if __name__ =='__main__':   
            cv2.imshow(f'contours_mask_with_boundingrect_%d'%(idx), contours_mask)
            cv2.waitKey()

    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)
    if defects is None:
        print("No Defects")
        return None, None, None
    else:
        print("defects shape:", defects.shape)
        #print("------defects-------------")
        #print(defects)
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])  
            far = tuple(cnt[f][0])
            #print("defect positoion:", start, end, far, d)
            cv2.line(convex_mask,start,end,[255-i*20,255-i*20,255],3)
            # if (d>100):
            #     cv2.circle(convex_mask,far,4,[255-i*20,255-i*20,255],-1)
            #     cv2.imshow(f'color_convex_mask_%d'%(idx), convex_mask)
            #     cv2.waitKey()

    # Morph close the convex hull mask, find contours, and fill in the outline
    convex_mask = cv2.cvtColor(convex_mask, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    convex_mask = cv2.morphologyEx(convex_mask, cv2.MORPH_CLOSE, kernel, iterations=10)
    cnts = cv2.findContours(convex_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cv2.fillPoly(convex_mask, cnts, (255,255,255))

    if __name__ =='__main__':
        cv2.imshow(f'convex_mask_%d'%(idx), convex_mask)
        cv2.waitKey()

    # Perform linear regression on the binary image
    [vx,vy,x,y] = cv2.fitLine(cnt,cv2.DIST_L2,0,0.01,0.01)
    lefty = int((-x*vy/vx) + y)
    righty = int(((image.shape[1]-x)*vy/vx)+y)
    cv2.line(line_mask,(image.shape[1]-1,righty),(0,lefty),[255,255,255],2)

    if __name__ =='__main__':
        cv2.imshow(f'%d_%d_%d_%d_line_mask_%d'%(image.shape[1]-1,righty,0,lefty,idx), line_mask)
        cv2.waitKey()

    #slope,intercept = np.polyfit([vx[0],x[0]], [vy[0],y[0]], 1)

    #print("slope, intercept ", slope, intercept)

    # slope,intercept = np.polyfit([image.shape[1]-1,righty], [0,lefty], 1)
    # angle = math.atan(slope) * 180/np.pi

    slope = (lefty-righty)/(0-(image.shape[1]-1))
    angle = math.atan(slope) * 180/np.pi
    # translate angles from topleft to bottom left refernce
    if (angle < 0):
        angle = -1 * angle
    else:
        angle = 180 - angle
    print("\n slope,angle", slope, angle,"\n")

    # Bitwise-and the line and convex hull masks together
    result = cv2.bitwise_and(line_mask, line_mask, mask=convex_mask)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    if __name__ =='__main__':
        cv2.imshow(f'result_%d'%(idx), result)
        cv2.waitKey()
        cv2.destroyAllWindows()

    # Find resulting contour and draw onto original image
    cnts = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]


    return cnts, cnt_length, angle

if __name__ == "__main__":
    args = sys.argv[1:]
    get_lanes(cv2.imread(args[0]), args[0], args[1])

# if __name__ == "__main__":
#     path = "data_2023/AI_data"
#     img_names = os.listdir(path)
#     for name in img_names:
#         img = cv2.imread(f"{path}/{name}")
#         get_lanes(img, name)