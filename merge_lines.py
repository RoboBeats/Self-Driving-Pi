import numpy as np
import math

def check_overlap(line1, line2):
    combination = np.array([line1,
                            line2,
                            [line1[0], line1[1], line2[0], line2[1]],
                            [line1[0], line1[1], line2[2], line2[3]],
                            [line1[2], line1[3], line2[0], line2[1]],
                            [line1[2], line1[3], line2[2], line2[3]]])
    distance = np.sqrt((combination[:,0] - combination[:,2])**2 + (combination[:,1] - combination[:,3])**2)
    max = np.amax(distance)
    overlap = distance[0] + distance[1] - max 
    endpoint = combination[np.argmax(distance)]
    return (overlap >= 0), endpoint #replace 0 with the value of distance between 2 collinear lines

def mergeLine(line_list, max_dist=10, max_angle=0.15):
    #convert (x1, y1, x2, y2) formm to (r, alpha) form
    A = line_list[:,1] - line_list[:,3]
    B = line_list[:,2] - line_list[:,0]
    C = line_list[:,0]*line_list[:,3] - line_list[:,2]*line_list[:,1]
    r = np.divide(np.abs(C), np.sqrt(A*A+B*B))
    alpha = (np.arctan2(-B,-A) + math.pi) % (2*math.pi) - math.pi
    r_alpha = np.column_stack((r, alpha))

    #prepare some variables to keep track of lines looping
    r_bin_size = max_dist #maximum distance to treat 2 lines as one
    alpha_bin_size = max_angle #maximum angle (radian) to treat 2 lines as one
    merged = np.zeros(len(r_alpha), dtype=np.uint8)
    line_group = np.empty((0,4), dtype=np.int32)
    group_count = 0

    for line_index in range(len(r_alpha)): 
        if merged[line_index] == 0: #if line hasn't been merged yet
            merged[line_index] = 1
            line_group = np.append(line_group, [line_list[line_index]], axis=0)
            for line_index2 in range(line_index+1,len(r_alpha)):
                if merged[line_index2] == 0:
                    #calculate the differences between 2 lines by r and alpha
                    dr = abs(r_alpha[line_index,0] - r_alpha[line_index2,0])
                    dalpha = abs(r_alpha[line_index,1] - r_alpha[line_index2,1])
                    if (dr<r_bin_size) and (dalpha<alpha_bin_size): #if they are close, they are the same line, so check if they are overlap
                        overlap, endpoints = check_overlap(line_group[group_count], line_list[line_index2])
                        if overlap:
                            line_group[group_count] = endpoints
                            merged[line_index2] = 1
            group_count += 1
    return line_group