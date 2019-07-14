import numpy as np
import cv2
import utility as u
import contour_helper as ch
import point_helper as ph

#Shortest distance between two line segments
def Dist_lineS_to_lineS (l0, l1):
    pt00=np.array([l0[0],l0[1]])
    pt01=np.array([l0[2],l0[3]])
    pt10=np.array([l1[0],l1[1]])
    pt11=np.array([l1[2],l1[3]])
    dist00=ph.Dist_pt_to_pt(pt00, pt10)
    dist01=ph.Dist_pt_to_pt(pt00, pt11)
    dist10=ph.Dist_pt_to_pt(pt01, pt10)
    dist11=ph.Dist_pt_to_pt(pt01, pt11)
    return min(dist00,dist01,dist10,dist11)

def HoughLinesP(input_image):
    padded_img = np.zeros((input_image.shape[0] + 2, 
                           input_image.shape[1] + 2), 
                           dtype=np.uint8)

    padded_img[1:-1, 1:-1] = input_image.copy()

    linesP = cv2.HoughLinesP(padded_img, rho = 1 , theta = np.pi/180,
                             threshold = 40, minLineLength = 10, maxLineGap = 50)

    return linesP

def Line_connection(l0, l1):
    y0 = float(l0[3] - l0[1])
    x0 = float(l0[2] - l0[0])
    angle0 = np.degrees(np.arctan2(y0, x0))

    y1 = float(l1[3] - l1[1])
    x1 = float(l1[2] - l1[0])
    angle1 = np.degrees(np.arctan2(y1, x1))

    #print "angles, ", angle0, angle1
#    print "angle ratio, dist", abs(angle0-angle1)/(abs(angle0)+abs(angle1)),  Dist_lineS_to_lineS(l0, l1)

    #if Dist_lineS_to_lineS(l0, l1) < 10 :
        #return True
    if abs(angle0-angle1)/(abs(angle0)+abs(angle1))<0.1 and Dist_lineS_to_lineS(l0, l1) < 60 :
        return True
    else:
        return False

# input_line_idx has to be the line closest to edge
def Lines_connections(input_line_idx, lines_input, connected_lines_indexes):
    l0 = lines_input[input_line_idx][0]
    
    if not Line_close_to_edge(l0):
        return []

    if not connected_lines_indexes:
        connected_lines_indexes.append(input_line_idx)

    if len(lines_input) < 1:
        return [input_line_idx]
    
    for comp_index in xrange (len(lines_input)):
        if comp_index == input_line_idx :
            continue
        if comp_index in connected_lines_indexes:
            continue

        #print "input %i and comparing with %i"%(input_line_idx, comp_index)
        l_comp = lines_input[comp_index][0]
        if Line_connection(l0, l_comp):
            connected_lines_indexes.append(comp_index)
            #print "start recursion with comp_index %i "%comp_index
            Lines_connections(comp_index, lines_input, connected_lines_indexes)
    
    return connected_lines_indexes

def Line_close_to_edge(l, dist=50):
    pt0 = np.array([l[0], l[1]])
    pt1 = np.array([l[2], l[3]])
    #print min(ph.Dist_pt_to_edge(pt0), ph.Dist_pt_to_edge(pt1))   
    if (ph.Dist_pt_to_edge(pt0) < dist) or (ph.Dist_pt_to_edge(pt1) < dist):
        #print "pt0", pt0
        #print "pt1", pt1

        return True
    else:
        return False



    
