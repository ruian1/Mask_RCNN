import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os, sys, gc, shutil
import pandas as pd
import ROOT
from larcv import larcv
import numpy as np
import tensorflow as tf

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)
sys.path.insert(0,os.path.join(BASE_PATH,".."))

from lib.config import config_loader
from lib.rootdata_maskrcnn import ROOTData

#for MCNN
#ROOT_DIR = os.path.abspath("../../../")
ROOT_DIR = os.path.abspath("/usr/local/share/dllee_unified/Mask_RCNN")
sys.path.append(ROOT_DIR)  #To find local version of the library

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.model import log
from mrcnn.config import Config
import mrcnn.model as modellib

BASE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(BASE_PATH)
sys.path.insert(0,BASE_PATH)
sys.path.insert(0,os.path.join(BASE_PATH,".."))

larcv.LArbysLoader()

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

def image_modify (img, cfg):
    img_arr = np.array(img.as_vector())
    img_arr = np.where(img_arr<cfg.adc_lo,         0,img_arr)
    img_arr = np.where(img_arr>cfg.adc_hi,cfg.adc_hi,img_arr)
    img_arr = img_arr.reshape(cfg.xdim,cfg.ydim, 1).astype(np.float32)
    
    return img_arr

'''
def nparray_modify(image_array, cfg):
    image_array = np.where(image_array<cfg.adc_lo,         0,image_array)
    image_array = np.where(image_array>cfg.adc_hi,cfg.adc_hi,image_array)
    image_array = image_array.reshape(cfg.xdim,cfg.ydim, 1).astype(np.float32)
    
    return image_array
'''

def nparray_modify(image_array, cfg):

    image_array = np.where(image_array<cfg.adc_lo,         0,image_array)
    image_array = np.where(image_array>cfg.adc_hi,cfg.adc_hi,image_array)

    if image_array.shape[0]==512:
        image_array = image_array.reshape(cfg.xdim,cfg.ydim, 1).astype(np.float32)
        return image_array

    result=np.zeros([512,512])
    
    input_len = image_array.shape[0]
    start = (512-input_len)/2
    end = 512 - start
    result[start:end, start:end] = image_array
    result = result.reshape(cfg.xdim,cfg.ydim, 1).astype(np.float32)
    
    return result




from MCNN_uboone_updated_debug import UbooneConfig
class InferenceConfig(UbooneConfig):
    #Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    #DETECTION_MAX_INSTANCES=10
    #RPN_ANCHOR_RATIOS = [0.125,0.5,1,2,4]
    DETECTION_MIN_CONFIDENCE = 0.6

def bb_too_small(bb):
    if ((bb[2]-bb[0]<10) and (bb[3]-bb[1])<10):
        return True
    else:
        return False

def merge_bbs(bbs):
    b=min([bb[0] for bb in bbs])
    t=max([bb[2] for bb in bbs])
    l=min([bb[1] for bb in bbs])
    r=max([bb[3] for bb in bbs])
    return [b,l,t,r]

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

def point_points_distances(point, points):
    points_x=points[0]
    points_y=points[1]
    point_x=point[0]
    point_y=point[1]
    distances=[((point_y-points_y[idx])**2+(point_x-points_x[idx])**2)**0.5 for idx in xrange(len(points_x))]
    if (len(distances)==0) : return -200
    return min(np.array(distances))

def main(IMAGE_FILE,VTX_FILE,OUT_DIR,CFG):

    class_names=[0, 11, -11, 13, -13, 22, 211, -211, 2212]

    cfg  = config_loader(CFG)
    assert cfg.batch == 1

    rd = ROOTData()

    NUM = int(os.path.basename(VTX_FILE).split(".")[0].split("_")[-1])
    FOUT = os.path.join(OUT_DIR,"maskrcnn_out_%04d.root" % NUM)
    #FOUT = os.path.join(OUT_DIR,"multimaskrcnn_out_04.root")
    tfile = ROOT.TFile.Open(FOUT,"RECREATE")
    tfile.cd()
    #print "OPEN %s"%FOUT

    tree  = ROOT.TTree("maskrcnn_tree","")
    rd.init_tree(tree)
    rd.reset()


    import MCNN_uboone_updated_debug

    config = UbooneConfig()

    #config.display()

    config = InferenceConfig()

    #
    #initialize iomanager
    #

    MODEL_DIR=OUT_DIR
    model = modellib.MaskRCNN(mode="inference", model_dir=OUT_DIR,config=config)

    #oiom = larcv.IOManager(larcv.IOManager.kWRITE)
    #oiom.set_out_file("trash.root")
    #oiom.initialize()

    iom  = larcv.IOManager(larcv.IOManager.kREAD)
    iom.add_in_file(IMAGE_FILE)
    iom.add_in_file(VTX_FILE)
    iom.initialize()

    #for entry in xrange(iom.get_n_entries()):
    for entry in xrange(1):

        iom.read_entry(entry)

        ev_pgr = iom.get_data(larcv.kProductPGraph,"inter_par")
        ev_par = iom.get_data(larcv.kProductPixel2D,"inter_par_pixel")
        ev_pix = iom.get_data(larcv.kProductPixel2D,"inter_img_pixel")
        ev_int = iom.get_data(larcv.kProductPixel2D,"inter_int_pixel")
        ev_img = iom.get_data(larcv.kProductImage2D,"wire")
        
        print '========================>>>>>>>>>>>>>>>>>>>>'
        print 'run, subrun, event',ev_pix.run(),ev_pix.subrun(),ev_pix.event()

        rd.run[0]    = int(ev_pix.run())
        rd.subrun[0] = int(ev_pix.subrun())
        rd.event[0]  = int(ev_pix.event())
        rd.entry[0]  = int(iom.current_entry())

        rd.num_vertex[0] = int(ev_pgr.PGraphArray().size())

        for ix,pgraph in enumerate(ev_pgr.PGraphArray()):
            #print "@pgid=%d" % ix
            #if (ix != 2): continue
            rd.vtxid[0] = int(ix)
   
            pgr = ev_pgr.PGraphArray().at(ix)
            cindex_v = np.array(pgr.ClusterIndexArray())
            
            #pixel2d_par_vv = ev_par.Pixel2DClusterArray()
            pixel2d_pix_vv = ev_pix.Pixel2DClusterArray()
            pixel2d_int_vv = ev_int.Pixel2DClusterArray()

            x,y,z=-1,-1,-1
            if (pgraph.ParticleArray().size()) :
                roi0 = pgraph.ParticleArray().front()
                x = roi0.X()
                y = roi0.Y()
                z = roi0.Z()
            
           
            for plane in xrange(3):
                if plane!=2 : continue
                if (pgraph.ParticleArray().size()) :

                    x_2d = ROOT.Double()
                    y_2d = ROOT.Double()
                    whole_img = ev_img.at(plane)
                    meta=pgraph.ParticleArray().front().BB()
                    

                    larcv.Project3D(meta[plane], x, y, z, 0.0, plane, x_2d, y_2d)
                    
                    width =int(meta[plane].tr().x-meta[plane].tl().x)
                    height=int(meta[plane].tl().y-meta[plane].br().y)
                    width = meta[plane].rows()
                    height = meta[plane].cols()


                    vertex_image=np.array(whole_img.crop(meta[plane]).as_vector()).reshape(width,height)

                    vertex_image_modified=nparray_modify(vertex_image,cfg)
                    
                    weight_file = ""
                    exec("weight_file = cfg.weight_file_mrcnn_plane%d" % plane)

                    model.load_weights(weight_file, by_name=True)
   

                    rd.inferred[0] = 1
                    '''
                    fig,ax=plt.subplots(1,1,figsize=(8,6))
                    print vertex_image_modified.shape
                    #ax.imshow(vertex_image_modified.reshape(cfg.xdim, cfg.ydim))
                    fig.savefig("%i_%i_%i_%i.pdf"%(ev_pix.run(),ev_pix.subrun(),ev_pix.event(),ix), bbox_inches='tight')
                    '''
                    
                    #Detection-1, for image with vectex centered
                    #from datetime import datetime
                    #a = datetime.now()
                
                    results_center = model.detect([vertex_image_modified], verbose=0)
                    
                    #b = datetime.now()
                    #c=b-a
                    #print 'using time of %i seconds'%c.seconds
                
                    r_center = results_center[0]


                    for each in r_center['scores']:
                        rd.center_scores_plane2.push_back(each)

                    for each in r_center['class_ids']:
                        rd.center_class_ids_plane2.push_back(class_names[each])
                            
                    for x in xrange(r_center['rois'].shape[0]):
                        roi_int=ROOT.std.vector("int")(4,0)
                        roi_int.clear()
                        for roi_int32 in r_center['rois'][x]:
                            roi_int.push_back(int(roi_int32))
                        rd.center_rois_plane2.push_back(roi_int)

                    classes_np=r_center['class_ids']
                    # masks are too large, now only store needed values
                    masks_np=np.zeros([r_center['masks'].shape[-1], cfg.xdim*cfg.ydim])

                    for x in xrange(r_center['masks'].shape[-1]):
                        this_mask=r_center['masks'][:,:,x]
                        this_mask=this_mask.flatten()
                        masks_np[x] = this_mask
                        '''
                        mask=ROOT.std.vector("bool")(cfg.xdim*cfg.ydim,False)
                        for idx in xrange(cfg.xdim*cfg.ydim):
                            mask[idx]=this_mask[idx]
                        rd.center_masks_plane2_1d.push_back(mask)
                        '''
                    idx=0
                    for each_class in classes_np :
                        pdg=class_names[each_class]
                        if pdg==11:
                            this_sum=np.sum(masks_np[idx])
                            rd.center_electron_mask_sum.push_back(np.int(this_sum))
                            this_dist=point_points_distances([256,256], np.nonzero(masks_np[idx].reshape(512,512)))
                            rd.center_electron_mask_dist.push_back(this_dist)
                        elif pdg==13:
                            this_sum=np.sum(masks_np[idx])
                            rd.center_muon_mask_sum.push_back(np.int(this_sum))
                            this_dist=point_points_distances([256,256], np.nonzero(masks_np[idx].reshape(512,512)))
                            rd.center_muon_mask_dist.push_back(this_dist)
                        elif pdg==2212:
                            this_sum=np.sum(masks_np[idx])
                            rd.center_proton_mask_sum.push_back(np.int(this_sum))
                            this_dist=point_points_distances([256,256], np.nonzero(masks_np[idx].reshape(512,512)))
                            rd.center_proton_mask_dist.push_back(this_dist)

                        idx+=1
                    #Store images in 2D vector, not compatible with pandas, uproot etc. lines above stored as 1d vector
                    '''
                    mask=ROOT.std.vector(ROOT.std.vector("bool"))(512, ROOT.std.vector("bool")(512, False))
                    this_mask=r_center['masks'][:,:,x]
                    for idx in xrange(this_mask.shape[0]):
                        for idy in xrange(this_mask.shape[1]):
                            mask[idx][idy]=this_mask[idx][idy]
                    rd.center_masks_plane2_2d.push_back(mask)
                    '''

                    

                    pixel2d_pix_v = pixel2d_pix_vv.at(plane)
                    pixel2d_pix = pixel2d_pix_v.at(ix)
                    
                    pixel2d_int_v = pixel2d_int_vv.at(plane)
                    pixel2d_int = pixel2d_int_v.at(ix)

                    img_pix = larcv.cluster_to_image2d(pixel2d_pix,cfg.xdim,cfg.ydim)
                    img_int = larcv.cluster_to_image2d(pixel2d_int,cfg.xdim,cfg.ydim)

                    img_pix_arr = image_modify(img_pix, cfg)
                    img_int_arr = image_modify(img_int, cfg)

                    #Detection 2(pixel image )
                    results_pix = model.detect([img_pix_arr], verbose=0)
                    
                    r_pix = results_pix[0]
                
                    for each in r_pix['scores']:
                        rd.pix_scores_plane2.push_back(each)

                    for each in r_pix['class_ids']:
                        rd.pix_class_ids_plane2.push_back(class_names[each])
                            
                    for x in xrange(r_pix['rois'].shape[0]):
                        roi_int=ROOT.std.vector("int")(4,0)
                        roi_int.clear()
                        for roi_int32 in r_pix['rois'][x]:
                            roi_int.push_back(int(roi_int32))
                        rd.pix_rois_plane2.push_back(roi_int)

                    classes_np=r_pix['class_ids']
                    # masks are too large, now only store needed values
                    masks_np=np.zeros([r_pix['masks'].shape[-1], cfg.xdim*cfg.ydim])
                    for x in xrange(r_pix['masks'].shape[-1]):
                        this_mask=r_pix['masks'][:,:,x]
                        this_mask=this_mask.flatten()
                        masks_np[x] = this_mask
                        '''
                        mask=ROOT.std.vector("bool")(cfg.xdim*cfg.ydim,False)
                        for idx in xrange(cfg.xdim*cfg.ydim):
                            mask[idx]=this_mask[idx]
                        rd.pix_masks_plane2_1d.push_back(mask)
                        '''

                    idx=0
                    for each_class in classes_np :
                        pdg=class_names[each_class]
                        if pdg==11:
                            this_sum=np.sum(masks_np[idx])
                            rd.pix_electron_mask_sum.push_back(np.int(this_sum))
                            this_dist=point_points_distances([256,256], np.nonzero(masks_np[idx].reshape(512,512)))
                            rd.pix_electron_mask_dist.push_back(this_dist)
                        elif pdg==13:
                            this_sum=np.sum(masks_np[idx])
                            rd.pix_muon_mask_sum.push_back(np.int(this_sum))
                            this_dist=point_points_distances([256,256], np.nonzero(masks_np[idx].reshape(512,512)))
                            rd.pix_muon_mask_dist.push_back(this_dist)
                        elif pdg==2212:
                            this_sum=np.sum(masks_np[idx])
                            rd.pix_proton_mask_sum.push_back(np.int(this_sum))
                            this_dist=point_points_distances([256,256], np.nonzero(masks_np[idx].reshape(512,512)))
                            rd.pix_proton_mask_dist.push_back(this_dist)

                        idx+=1

                    #Detection 3(interaction)
                    results_int = model.detect([img_int_arr], verbose=0)
                    
                    r_int = results_int[0]
                
                    for each in r_int['scores']:
                        rd.int_scores_plane2.push_back(each)

                    for each in r_int['class_ids']:
                        rd.int_class_ids_plane2.push_back(class_names[each])
                            
                    for x in xrange(r_int['rois'].shape[0]):
                        roi_int=ROOT.std.vector("int")(4,0)
                        roi_int.clear()
                        for roi_int32 in r_int['rois'][x]:
                            roi_int.push_back(int(roi_int32))
                        rd.int_rois_plane2.push_back(roi_int)

                    classes_np=r_int['class_ids']
                    # masks are too large, now only store needed values
                    masks_np=np.zeros([r_int['masks'].shape[-1], cfg.xdim*cfg.ydim])
                    for x in xrange(r_int['masks'].shape[-1]):
                        this_mask=r_int['masks'][:,:,x]
                        this_mask=this_mask.flatten()
                        masks_np[x] = this_mask
                        '''
                        mask=ROOT.std.vector("bool")(cfg.xdim*cfg.ydim,False)
                        for idx in xrange(cfg.xdim*cfg.ydim):
                            mask[idx]=this_mask[idx]
                        rd.int_masks_plane2_1d.push_back(mask)
                        '''

                    idx=0
                    for each_class in classes_np :
                        pdg=class_names[each_class]
                        if pdg==11:
                            this_sum=np.sum(masks_np[idx])
                            rd.int_electron_mask_sum.push_back(np.int(this_sum))
                            this_dist=point_points_distances([256,256], np.nonzero(masks_np[idx].reshape(512,512)))
                            rd.int_electron_mask_dist.push_back(this_dist)
                        elif pdg==13:
                            this_sum=np.sum(masks_np[idx])
                            rd.int_muon_mask_sum.push_back(np.int(this_sum))
                            this_dist=point_points_distances([256,256], np.nonzero(masks_np[idx].reshape(512,512)))
                            rd.int_muon_mask_dist.push_back(this_dist)
                        elif pdg==2212:
                            this_sum=np.sum(masks_np[idx])
                            rd.int_proton_mask_sum.push_back(np.int(this_sum))
                            this_dist=point_points_distances([256,256], np.nonzero(masks_np[idx].reshape(512,512)))
                            rd.int_proton_mask_dist.push_back(this_dist)


                        idx+=1
                    
                    fig,(ax0, ax1, ax2)=plt.subplots(1,3,figsize=(21,7))
                    visualize.display_instances(vertex_image_modified, r_center['rois'],
                                                r_center['masks'], r_center['class_ids'],
                                                class_names, r_center['scores'], ax=ax0,
                                                title="center_Predictions")
                    visualize.display_instances(img_pix_arr, r_pix['rois'],
                                                r_pix['masks'], r_pix['class_ids'],
                                                class_names, r_pix['scores'], ax=ax1,
                                                title="pix_Predictions")
                    visualize.display_instances(img_int_arr, r_int['rois'],
                                                r_int['masks'], r_int['class_ids'],
                                                class_names, r_int['scores'], ax=ax2,
                                                title="int_Predictions")
                    fig.savefig("%i_%i_%i_%i.pdf"%(ev_pix.run(),ev_pix.subrun(),ev_pix.event(),ix), bbox_inches='tight')
                    


            tree.Fill()
            rd.reset_vertex()
    tfile.cd()
    tree.Write()
    tfile.Close()
    iom.finalize()

if __name__ == '__main__':
    
    if len(sys.argv) != 5:
        print
        print "\tIMAGE_FILE = str(sys.argv[1])"
        print "\tVTX_FILE   = str(sys.argv[2])"
        print "\tOUT_DIR    = str(sys.argv[3])"
        print "\tCFG        = str(sys.argv[4])"
        print 
        sys.exit(1)
    
    IMAGE_FILE = str(sys.argv[1]) 
    VTX_FILE   = str(sys.argv[2])
    OUT_DIR    = str(sys.argv[3])
    CFG        = str(sys.argv[4])

    #CFG = os.path.join(BASE_PATH,"cfg","simple_config.cfg")

    with tf.device('/cpu:0'):
        main(IMAGE_FILE,VTX_FILE,OUT_DIR,CFG)
    
    print "DONE!"
    sys.exit(0)
