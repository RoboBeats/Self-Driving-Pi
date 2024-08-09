from merge_linesv2 import *
import numpy as np

line1 = [478.0, 231.0, 482.0, 380.0]
line2 = [489.0, 283.0, 493.0, 329.0]
p1, p2 = line1[:2], line1[2:]
p3, p4 = line2[:2], line2[2:]
breakpoint()
a = merge(np.array([line1, line2]), 100,20, im_shape=np.zeros((480, 640, 3)))
print(a)