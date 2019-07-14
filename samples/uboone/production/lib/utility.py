from larcv import larcv
import ROOT
import numpy as np
import cv2

def Calculate_edge_pt (contour, edge):
    # to bottom
    if edge == 0:
        dist_v = contour[:,1]
        return contour[np.argmin(dist_v)]
    # to up
    if edge == 1:
        dist_v = 512 - contour[:,1]
        return contour[np.argmin(dist_v)]
    # to left
    if edge == 2:
        dist_v = contour[:,0]
        return contour[np.argmin(dist_v)]
    # to right
    if edge == 3:
        dist_v = 512 - contour[:,0]
        return contour[np.argmin(dist_v)]

def Calculate_initial_pt (contour, edge):
    # to bottom
    if edge == 0:
        dist_v = contour[:,1]
        return contour[np.argmax(dist_v)]
    # to up
    if edge == 1:
        dist_v = 512 - contour[:,1]
        return contour[np.argmax(dist_v)]
    # to left
    if edge == 2:
        dist_v = contour[:,0]
        return contour[np.argmax(dist_v)]
    # to right
    if edge == 3:
        dist_v = 512 - contour[:,0]
        return contour[np.argmax(dist_v)]

def Calculate_contour_edge (contour):
    pt = [Calculate_contour_mean_x(contour), Calculate_contour_mean_y(contour)]

    dist, side = Pt_dist_to_bottom (pt)
    if Pt_dist_to_up (pt)[0] < dist : 
        dist, side = Pt_dist_to_up (pt)
    if Pt_dist_to_left (pt)[0] < dist : 
        dist, side = Pt_dist_to_left (pt)
    if Pt_dist_to_right (pt)[0] < dist : 
        dist, side = Pt_dist_to_right (pt)
    return side

def Find_contours(img):
    padded_img = np.zeros((img.shape[0] + 2, img.shape[1] + 2), dtype=np.uint8)
    padded_img[1:-1, 1:-1] = img
    if cv2.__version__=="3.0.0" :
        im2, contours, hierarchy = cv2.findContours(padded_img.copy(),
                                                     cv2.RETR_TREE,
                                                     cv2.CHAIN_APPROX_NONE)
    if cv2.__version__=="4.0.0" :
        contours, hierarchy = cv2.findContours(padded_img.copy(),
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_NONE)
    
    contours=np.array(contours)
    contours=contours-1
    
    return img, contours, hierarchy

#Get the 512x512 cropped image
def Get_crop_image(plane, x_2d, y_2d, ev_img):
    meta_crop = larcv.ImageMeta(512,512*6,512,512,0,8448,plane)   
    meta_origin_x, meta_origin_y = meta_origin_helper(x_2d, y_2d, verbose=1)
    meta_crop.reset_origin(meta_origin_x, meta_origin_y)
    img_vtx = ev_img.at(plane).crop(meta_crop)
    img_vtx = larcv.as_ndarray(img_vtx)
    img_vtx = np.where(img_vtx<10, 0,img_vtx)
    img_vtx = np.where(img_vtx>500, 500,img_vtx)
    return img_vtx

def image_modify (img):
    img_arr = np.array(img.as_vector())
    img_arr = np.where(img_arr<10,         0,img_arr)
    img_arr = np.where(img_arr>500,      500,img_arr)
    img_arr = img_arr.reshape(1,img_arr.size).astype(np.float32)

    return img_arr

def Meta_origin_helper (x_2d , y_2d , get_new_origin=False, verbose=False, get_shifts=False):
    x_shift = 0
    y_shift = 0
    meta_origin_x = x_2d-256
    if meta_origin_x < 0 : 
        x_shift = 0 - meta_origin_x
        if (verbose): print "case#1 meta_ori x < 0, shifting x by ",x_shift
        meta_origin_x = 0
    if meta_origin_x > (3456 - 512):
        x_shift = - meta_origin_x + (3456-512)
        if (verbose): print "case#2 meta_ori x > 3456-512, shifting x by", x_shift
        meta_origin_x = 3456-512

    meta_origin_y = (1008 - y_2d)*6 +2400 + 256 * 6
    if meta_origin_y < 2400+512*6 : 
        y_shift = 2400+512*6 - meta_origin_y
        if (verbose): print "case#3 meta_ori y < 2400 + 512*6, shifting y by ", y_shift
        meta_origin_y = 2400+512*6
    if meta_origin_y > 8448 :
        y_shift = - meta_origin_y + 8448
        if (verbose): print "case#4 meta_ori y > 8448, shifting y by ", y_shift
        meta_origin_y = 8448
        
    new_y_2d = 256 + y_shift/6
    new_x_2d = 256 - x_shift
    if (not get_new_origin):
        return meta_origin_x, meta_origin_y
    if (get_new_origin):
        return new_y_2d , new_x_2d
    if (get_shifts):
        return x_shift, y_shift

def Nparray_modify(image_array):

    image_array = np.where(image_array<10,    0, image_array)
    image_array = np.where(image_array>500, 500, image_array)

    
    return image_array.reshape(512, 512, 1).astype(np.float64)

    '''
    result=np.zeros([512,512])
    
    input_len = image_array.shape[0]
    start = (512-input_len)/2
    end = 512 - start
    result[start:end, start:end] = image_array
    result = result.reshape(512, 512, 1).astype(np.float32)
    
    return result
    '''
def Pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


#Project 3D pt to 2D plane and flip to fit time tick
def Project3Dto2D_and_fliplr(meta, x, y, z, plane, x_2d, y_2d):
    larcv.Project3D(meta, x, y, z, 0.0, plane, x_2d, y_2d)
    return x_2d, 1008-y_2d
