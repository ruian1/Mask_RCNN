import numpy as np

def BB_too_small(bb):
    if ((bb[2]-bb[0]<10) and (bb[3]-bb[1])<10):
        return True
    else:
        return False

def IOU(bbs_t, bbs_r):
    bbs_t=merge_bbs(bbs_t)
    bbs_r=merge_bbs(bbs_r)
    tot_b=min(bbs_t[0], bbs_r[0])
    tot_t=max(bbs_t[2], bbs_r[2])
    tot_l=min(bbs_t[1], bbs_r[1])
    tot_r=max(bbs_t[3], bbs_r[3])

    int_b=max(bbs_t[0], bbs_r[0])
    int_t=min(bbs_t[2], bbs_r[2])
    int_l=max(bbs_t[1], bbs_r[1])
    int_r=min(bbs_t[3], bbs_r[3])
    
    tot_area=(tot_r-tot_l)*(tot_t-tot_b)
    int_area=(int_r-int_l)*(int_t-int_b)

    return(int_area)/(tot_area)

def Merge_bbs(bbs):
    b=min([bb[0] for bb in bbs])
    t=max([bb[2] for bb in bbs])
    l=min([bb[1] for bb in bbs])
    r=max([bb[3] for bb in bbs])
    return [b,l,t,r]

def Point_points_distances(point, points):
    points_x=points[0]
    points_y=points[1]
    point_x=point[0]
    point_y=point[1]
    distances=[((point_y-points_y[idx])**2+(point_x-points_x[idx])**2)**0.5 for idx in xrange(len(points_x))]
    if (len(distances)==0) : return -200
    return min(np.array(distances))

