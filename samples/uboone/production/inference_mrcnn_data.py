import os, sys, gc
import ROOT
from larcv import larcv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

#for MCNN
ROOT_DIR = os.path.abspath("../../../")
sys.path.append(ROOT_DIR)  # To find local version of the library

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

from lib.config import config_loader
from lib.rootdata_pid import ROOTData

larcv.LArbysLoader()


p_type = {0:"eminus", 1:"gamma", 2:"muon", 3:"piminus",4:"proton"}

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector

def image_modify (img):
    img_arr = np.array(img.as_vector())
    img_arr = np.where(img_arr<10,0,img_arr)
    #img_arr = np.where(img_arr>cfg.adc_hi,cfg.adc_hi,img_arr)
    img_arr = img_arr.reshape(1,img_arr.size).astype(np.float32)
    return img_arr
    
# Override the training configurations with a few
# changes for inferencing.
from MCNN_uboone import UbooneConfig
class InferenceConfig(UbooneConfig):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    #DETECTION_MAX_INSTANCES=10
    #RPN_ANCHOR_RATIOS = [0.125,0.5,1,2,4]
    #DETECTION_MIN_CONFIDENCE = 0.8

def main(INPUT_FILE,OUT_DIR,CFG):
    
 
    #
    # initialize Mask RCNN
    #    
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
    cfg  = config_loader(CFG)
    assert cfg.batch == 1
    
    import MCNN_uboone

    config = UbooneConfig()

    # config.display()

    config = InferenceConfig()

    MODEL_DIR="/scratch/ruian/maskrcnn_log"
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,config=config)
    plane=cfg.plane
    
    weights_path=cfg.weight_mrcnn_plane2
    model.load_weights(weights_path, by_name=True)
    
    #
    #initialized ROOT
    #
    rd = ROOTData()

    #NUM = int(os.path.basename(VTX_FILE).split(".")[0].split("_")[-1])
    NUM=123
    FOUT = os.path.join(OUT_DIR,"multipid_out_%d.root" % NUM)
    #FOUT = os.path.join(OUT_DIR,"multipid_out_04.root")
    tfile = ROOT.TFile.Open(FOUT,"RECREATE")
    tfile.cd()
    print "OPEN %s"%FOUT

    tree  = ROOT.TTree("multipid_tree","")
    rd.init_tree(tree)
    rd.reset()
    
    #
    # initialize iomanager
    #

    # oiom = larcv.IOManager(larcv.IOManager.kWRITE)
    # oiom.set_out_file("trash.root")
    # oiom.initialize()
    iom  = larcv.IOManager(larcv.IOManager.kREAD)
    iom.add_in_file(INPUT_FILE)
    #iom.add_in_file(VTX_FILE)
    iom.initialize()


    #
    # running detection
    #
    #for entry in xrange(iom.get_n_entries()):
    for entry in xrange(20):
        iom.read_entry(entry)
        
        ev_pgr = iom.get_data(larcv.kProductPGraph,"inter_par")
        ev_pix_int = iom.get_data(larcv.kProductPixel2D,"inter_int_pixel")
        ev_pix_pix = iom.get_data(larcv.kProductPixel2D,"inter_img_pixel")
        
        #if not (ev_pix_int.run()==RUN and ev_pix_int.subrun()==SUBRUN and  ev_pix_int.event()==EVENT):continue
        print '>>>>>>>>'
        print 'run, subrun, event',ev_pix_int.run(),ev_pix_int.subrun(),ev_pix_int.event()
        print ev_pgr.PGraphArray().size()
        
        for ix,pgraph in enumerate(ev_pgr.PGraphArray()):
            #if not (ix == VTXID): continue
            if (not ev_pgr.PGraphArray().size()) : continue
                    
            print "@pgid=%d" % ix
            
            pixel2d_vv_int = ev_pix_int.Pixel2DClusterArray()
            pixel2d_vv_pix = ev_pix_pix.Pixel2DClusterArray()
            cluster_vv_int = ev_pix_int.ClusterMetaArray()

            print 'int size ', ev_pix_int.Pixel2DClusterArray().size()
            print 'pix size ', ev_pix_pix.Pixel2DClusterArray().size()
            print 'int size ', ev_pix_int.ClusterMetaArray().size()
            print 'There are ', pgraph.ParticleArray().size(), 'clusters.'

            if (pgraph.ParticleArray().size()==0):
                continue
            
            parid = pgraph.ClusterIndexArray().front()
            roi0 = pgraph.ParticleArray().front()
            x = roi0.X()
            y = roi0.Y()
            z = roi0.Z()
            
            y_2d_plane_0 = ROOT.Double()
            
            for plane in xrange(3):
                if plane == 0: continue
                if plane == 1: continue
                
                print "@plane=%d" % plane
                
                if pixel2d_vv_int.empty()==True: continue
                '''
                pixel2d_v_int = pixel2d_vv_int.at(plane)
                pixel2d_int   = pixel2d_v_int.at(ix)
                img_int = larcv.cluster_to_image2d(pixel2d_int,512, 512)
                img_arr_int = image_modify(img_int)
                img_arr_=img_arr_int.reshape(512,512)
                '''
                pixel2d_v_pix = pixel2d_vv_pix.at(plane)
                pixel2d_pix   = pixel2d_v_pix.at(ix)
                
                img_pix = larcv.cluster_to_image2d(pixel2d_pix,512, 512)
                img_arr_pix = image_modify(img_pix)
                
                img_arr_=img_arr_pix.reshape(512,512)
                
                input_image=img_arr_
                
                
                results = model.detect([input_image.reshape(512,512,1)], verbose=1)
                r = results[0]
                print 'r is ', 
                print r
    tfile.cd()
    tree.Write()
    tfile.Close()
    iom.finalize()
    #oiom.finalize()

if __name__ == '__main__':
    
    if len(sys.argv) != 3:
        print
        print "\tINPUT_FILE = str(sys.argv[1])"
        #print "\tVTX_FILE = str(sys.argv[2])"
        print "\tOUT_DIR  = str(sys.argv[2])"
        #print "\tRUN = int(sys.argv[4])"
        #print "\tSUBRUN = int(sys.argv[5])"
        #print "\tEVENT = int(sys.argv[6])"
        #print "\tVTXID = int(sys.argv[7])"

        print 
        sys.exit(1)
    
    INPUT_FILE = str(sys.argv[1]) 
    #VTX_FILE = str(sys.argv[2])
    OUT_DIR  = str(sys.argv[2])
    #RUN = int(sys.argv[4])
    #SUBRUN = int(sys.argv[5])    
    #EVENT = int(sys.argv[6])
    #VTXID = int(sys.argv[7])
    
    CFG = os.path.join(BASE_PATH,"cfg","simple_config.cfg")

    main(INPUT_FILE,OUT_DIR,CFG)
    
    sys.exit(0)


