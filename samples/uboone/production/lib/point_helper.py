import numpy as np
import cv2
import utility as u
import contour_helper as ch
import line_helper as lh

#Shortest distance from 1 pt to cloest edge
def Dist_pt_to_edge(pt):
    pt_x_min = np.array([0, pt[1]])
    pt_x_max = np.array([512, pt[1]])
    pt_y_min = np.array([pt[0], 0])
    pt_y_max = np.array([pt[0], 512])
    
    return min(Dist_pt_to_pt(pt, pt_x_min),Dist_pt_to_pt(pt, pt_x_max),
               Dist_pt_to_pt(pt, pt_y_min),Dist_pt_to_pt(pt, pt_y_max))

def Dist_pt_to_pt (pt1, pt2):
    return np.sqrt(np.sum(np.power(pt2-pt1, 2)))

def Dist_pt_to_bottom (pt):
    return pt[1], 0

def Dist_pt_to_up (pt):
    return 512 -  pt[1], 1

def Dist_pt_to_left (pt):
    return pt[0], 2

def Dist_pt_to_right (pt):
    return 512 - pt[0], 3

