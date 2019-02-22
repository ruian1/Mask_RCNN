import os, sys, gc, shutil
import pandas as pd
import ROOT
from larcv import larcv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

from MCNN_uboone import UbooneConfig
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
    print "OPEN %s"%FOUT

    tree  = ROOT.TTree("maskrcnn_tree","")
    rd.init_tree(tree)
    rd.reset()


    import MCNN_uboone

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

    for entry in xrange(iom.get_n_entries()):

        iom.read_entry(entry)

        ev_pgr = iom.get_data(larcv.kProductPGraph,"inter_par")
        ev_par = iom.get_data(larcv.kProductPixel2D,"inter_par_pixel")
        ev_pix = iom.get_data(larcv.kProductPixel2D,"inter_img_pixel")
        ev_int = iom.get_data(larcv.kProductPixel2D,"inter_int_pixel")
        #ev_img = iom.get_data(larcv.kProductImage2D,"wire")
        
        #print '========================>>>>>>>>>>>>>>>>>>>>'
        #print 'run, subrun, event',ev_pix.run(),ev_pix.subrun(),ev_pix.event()

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

            #parid = pgraph.ClusterIndexArray().front()
            #roi0 = pgraph.ParticleArray().front()

            #x = roi0.X()
            #y = roi0.Y()
            #z = roi0.Z()

            #y_2d_plane_0 = ROOT.Double()

            for plane in xrange(3):
                if plane!=2 : continue
                #print "@plane=%d" % plane
                
                ###Get 2D vertex Image
                
                #meta = roi0.BB(plane)

                #x_2d = ROOT.Double()
                #y_2d = ROOT.Double()
                
                #whole_img = ev_img.at(plane)
                
                #larcv.Project3D(whole_img.meta(), x, y, z, 0.0, plane, x_2d, y_2d)
                ##print 'x2d, ', x_2d, 'y2d, ',y_2d
                
                #if (plane == 0) : y_2d_plane_0 = y_2d
                #else : y_2d = y_2d_plane_0
                
                ###
                weight_file = ""
                exec("weight_file = cfg.weight_file_mrcnn_plane%d" % plane)

                model.load_weights(weight_file, by_name=True)
                
                #nothing
                #if pixel2d_vv.empty()==True: continue

                pixel2d_pix_v = pixel2d_pix_vv.at(plane)
                pixel2d_pix = pixel2d_pix_v.at(ix)

		pixel2d_int_v = pixel2d_int_vv.at(plane)
                pixel2d_int = pixel2d_int_v.at(ix)
            
                #nothing on this plane
                #if pixel2d_pix.empty() == True: continue

                rd.inferred[0] = 1
                
                img_pix = larcv.cluster_to_image2d(pixel2d_pix,cfg.xdim,cfg.ydim)
                img_int = larcv.cluster_to_image2d(pixel2d_int,cfg.xdim,cfg.ydim)

                img_pix_arr = image_modify(img_pix, cfg)
                img_int_arr = image_modify(img_int, cfg)
                
                #Detection
                #from datetime import datetime
                #a = datetime.now()
                
                results = model.detect([img_pix_arr], verbose=0)

                #b = datetime.now()
                #c=b-a
                #print 'using time of %i seconds'%c.seconds

                r = results[0]
                
                for each in r['scores']:
                    rd.scores_plane2.push_back(each)

                for each in r['class_ids']:
                    rd.class_ids_plane2.push_back(class_names[each])

                for x in xrange(r['rois'].shape[0]):
                    roi_int=ROOT.std.vector("int")(4,0)
                    roi_int.clear()
                    for roi_int32 in r['rois'][x]:
                        roi_int.push_back(int(roi_int32))
                    rd.rois_plane2.push_back(roi_int)

                for x in xrange(r['masks'].shape[-1]):
                    mask=ROOT.std.vector(ROOT.std.vector("bool"))(512, ROOT.std.vector("bool")(512, False))
                    this_mask=r['masks'][:,:,x]
                    for idx in xrange(this_mask.shape[0]):
                        for idy in xrange(this_mask.shape[1]):
                            mask[idx][idy]=this_mask[idx][idy]
                    rd.masks_plane2.push_back(mask)
                    
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
