import numpy as np
import cv2
import utility as u
import line_helper as lh
import point_helper as ph


#Contour mean in x direction
def Calculate_contour_mean_x (contour):
    return np.mean(contour[:,0])

#Contour mean in y direction
def Calculate_contour_mean_y (contour):
    return np.mean(contour[:,1])

#Check if contour is close to edge
def If_contour_to_edge(contour, dist=5):
    for pt_c in contour:
        if dist_pt_to_edge(pt_c) < dist:
            return True
    return False     
