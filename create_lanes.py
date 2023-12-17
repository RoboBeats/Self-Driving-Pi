import cv2
import sys
import numpy as np
import math
from lane_func import get_votes, fit_and_angle, cnt_is_blocking, check_pair


MIN_LANE_LENGTH = 500
MAX_LANE_WIDTH = 50        #Max lane marker width, if more than that some other object
LANE_COLOR = (200,100,100)
POLY_CONTOUR_COLOR = (50,50,50)
NON_LANE_COLOR = (100,100,100)
MIN_BLOCK_AREA = 75000

def delete_top(image):
    shape = image.shape
    height = shape[0]
    if len(shape) == 2:
        resized_image = np.delete(image, slice(int(height/3)),0)
    else:
        resized_image = np.delete(image, slice(int(height/3)),0)
    print('input image dimension:', image.shape, '  resized image dimensions:', resized_image.shape)
    return resized_image

def get_lanes(im_name, save_dir=""):
    # Load image, delete top portion, grayscale, Otsu's threshold
    original_image = cv2.imread(im_name)
    image = delete_top(original_image)

    #sharpening_kernel =np.array( [[-1, -1, -1],[-1,9,-1], [-1,-1,-1]])
    #sharpened = cv2.filter2D(image, -1, sharpening_kernel)
    #gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if __name__ =='__main__':
        cv2.imshow('original',original_image)
        cv2.imshow('gray_resized', gray)
        cv2.waitKey()

    
    #ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    print("ret for thrresh:", ret)
    if __name__ =='__main__':
        cv2.imshow('thresh', thresh)
        cv2.waitKey()

    # edges = cv2.Canny(gray, 127, 300, L2gradient =True)
    # if __name__ =='__main__':
    #     cv2.imshow('Canny', edges)
    #     cv2.waitKey()
   

    # Morph open to remove noise, then Morph close to fill in 
    noise_removal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, noise_removal_kernel)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, close_kernel, iterations=3)
    if __name__ =='__main__':
        cv2.imshow('close', close)
        cv2.waitKey()

    # Find contours
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("contours.len:", len(cnts))
    #print("-----------Contours[0]------------------")
    #print(cnts[0])
    print("-----------Contours[1]------------------")
    print("cnt1 shape:", cnts[1].shape)
    print(cnts[1])

    hierarchy = cnts[1]
    contours = cnts[0]

    #cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    lanes = []
    print("Num contours:",hierarchy.shape[1])

    #Process each contour to get the bounding box and identity object if possible
    for idx in range(hierarchy.shape[1]):
        print("for loop idx",idx)
        cnt = contours[idx]
        
        # print("\n\nContour: \n", cnt, "\n___________________________________\n")
        
        # check if lane
        votes, cnt_len = get_votes(image, cnt, idx, __name__ =='__main__')
        if votes>0:
            slope, intercept, angle, lane_points = fit_and_angle(cnt, image, __name__ =='__main__')
            lanes.append([cnt, votes, cnt_len, slope, intercept, angle, lane_points, idx])

        #print("lanes:", lanes)
        #sort lanes array by lane length

    lanes = sorted(lanes, key=lambda x: (x[1], x[2]), reverse=True)
    final_lanes = lanes[:2]
    """
    if len(lanes) > 1:
        pairs = []
        for i in range(len(lanes)):
            for j in range(i+1, len(lanes)):
                is_pair = check_pair(lanes[i], lanes[j], image.shape[1], image.shape[0])
                pairs.append([i, j, is_pair])
        pairs = sorted(pairs, key=lambda x: (x[2], lanes[x[0]][1], lanes[x[1]][1], lanes[x[0]][2]), reverse=True)
        print("_________________\n pairs:", pairs, "\n_________________\n")
        if pairs[0][2] == 0:
            final_lanes = [lanes[0]]
        else:
            final_lanes = [lanes[pairs[0][0]], lanes[pairs[0][1]]]

    elif len(lanes) == 1:
        final_lanes = [lanes[0]]
    else:
        return [], True
    """

    print("\n final lanes: ", len(final_lanes))
    for idx in range(hierarchy.shape[1]):
        cnt = contours[idx]
        cnt_rect = cv2.minAreaRect(cnt)
        area = cnt_rect[1][0]*cnt_rect[1][1]
        if area < MIN_BLOCK_AREA:
            continue

        print("cnt area, idx:  ", area, idx)
        is_lane = False
        for lane in final_lanes:
            print("CHECKING FOR LANE")
            if idx == lane[7]:
                print("THIS IS A LANE")
                is_lane = True
                break
        if is_lane:
            continue

        print("Checking for Obstruction")
        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        stop = True
        for lane in final_lanes:
            _, _, _, slope, intercept, angle, _, _ = lane
            stop = stop and cnt_is_blocking(leftmost, rightmost, slope, intercept, angle)
            print("stop in for loop: ", stop)
        
        if stop == True:
            return final_lanes, True

    return final_lanes, False

def display_lanes(image, lanes, im_name, save_dir):
    #Draw lanes on image and display
    num_lanes=0
    for lane in lanes:
        cv2.drawContours(image, lane[0], -1, LANE_COLOR, 20)
        # x,y,w,h = lane[2]
        # print(x,y,w,h)
        if len(save_dir) != 0:
            cv2.imwrite(save_dir+im_name, image)

        #cv2.drawContours(image,lane[4], -1, POLY_CONTOUR_COLOR, 10)
        #cv2.rectangle(image, (x,y), (x+w, y+h), LANE_COLOR, 3)
        num_lanes+= 1
        if num_lanes ==2:
            break
    
    if __name__ =='__main__':
        cv2.imshow('image_with_lanes', image)    
        cv2.waitKey()

""" 
    Checks if the input contour is a valid lane, and if so returns lane contours, slope, and intercept.
    Returns None,None,None otherwise
"""

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

    # Find resulting contour and draw onto original image
    cnts = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]


    return cnts, cnt_length, angle

if __name__ == "__main__":
    args = sys.argv[1:]
    _, stop = get_lanes(args[0])
    print(f'\n\n-------------{stop}-----------------\n')