import os, sys, gc
import pandas as pd
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

def main(INPUT_FILE,OUT_DIR,CFG):
    
 
    #
    # initialize Mask RCNN
    #    

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    
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

    weight_file_name = weights_path.split('/')[-1]
    '''
    fout = open('%s.csv' % (weight_file_name),'w')
    #fout.write('run,subrun,event,')
    fout.write('true_n_em,true_n_ga,true_n_mu,true_n_pi,true_n_pr,')
    fout.write('em_bboxes_t,ga_bboxes_t,mu_bboxes_t,pi_bboxes_t,pr_bboxes_t,')
    fout.write('reco_n_em,reco_n_ga,reco_n_mu,reco_n_pi,reco_n_pr,')
    fout.write('em_scores,ga_scores,mu_scores,pi_scores,pr_scores,')
    fout.write('em_bboxes_r,ga_bboxes_r,mu_bboxes_r,pi_bboxes_r,pr_bboxes_r,')
    fout.write('em_iou,ga_iou,mu_iou,pi_iou,pr_iou')
    fout.write('\n')
    '''
    col_names =  ['true_n_em','true_n_ga','true_n_mu','true_n_pi','true_n_pr',\
                  'reco_n_em','reco_n_ga','reco_n_mu','reco_n_pi','reco_n_pr',\
                  'em_bboxes_t','ga_bboxes_t','mu_bboxes_t','pi_bboxes_t','pr_bboxes_t',\
                  'em_bboxes_r','ga_bboxes_r','mu_bboxes_r','pi_bboxes_r','pr_bboxes_r',\
                  'em_scores','ga_scores','mu_scores','pi_scores','pr_scores',\
                  'em_scores_mean','ga_scores_mean','mu_scores_mean','pi_scores_mean','pr_scores_mean',\
                  'em_iou','ga_iou','mu_iou','pi_iou','pr_iou']
    outdf  = pd.DataFrame(columns = col_names)
    #outdf=outdf.astype('float')

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
    #print "OPEN %s"%FOUT

    tree  = ROOT.TTree("multipid_tree","")
    rd.init_tree(tree)
    rd.reset()

    #
    #Read MC data
    #
    dataset = MCNN_uboone.UbooneDataset(INPUT_FILE)
    dataset.load_events(1990, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset.prepare()
    
    #
    # running detection
    #
    for entry in xrange(len(dataset.image_info)):
        #truth info
        print entry
        image_id=entry
        image, image_meta, gt_class_id, gt_bboxes, gt_mask =\
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        info = dataset.image_info[image_id]

        true_pdgs=[dataset.class_names[x]  for x in gt_class_id ]

        true_n_em=true_pdgs.count(11)
        true_n_ga=true_pdgs.count(22)
        true_n_mu=true_pdgs.count(13)
        true_n_pi=true_pdgs.count(211)
        true_n_pr=true_pdgs.count(2212)
        '''
        print '>>>>>>>>>>>>>>>>>>>'
        print 'gt_class_id is '
        print gt_class_id
        print [dataset.class_names[x]  for x in gt_class_id ]
        print 'gt_bboxes is '
        print  gt_bboxes
        '''
        em_bboxes_t=[]
        ga_bboxes_t=[]
        mu_bboxes_t=[]
        pi_bboxes_t=[]
        pr_bboxes_t=[]

        assert len(true_pdgs)==len(gt_bboxes), 'True pdg and bbox length are NOT same!!!'
        
        for idx in xrange(len(true_pdgs)):
            pdg=true_pdgs[idx]
            if pdg==11   : em_bboxes_t.append(gt_bboxes[idx])
            if pdg==22   : ga_bboxes_t.append(gt_bboxes[idx])
            if pdg==13   : mu_bboxes_t.append(gt_bboxes[idx])
            if pdg==211  : pi_bboxes_t.append(gt_bboxes[idx])
            if pdg==2212 : pr_bboxes_t.append(gt_bboxes[idx])

        '''
        fout.write('%i,'%true_n_em)
        fout.write('%i,'%true_n_ga)
        fout.write('%i,'%true_n_mu)
        fout.write('%i,'%true_n_pi)
        fout.write('%i,'%true_n_pr)

        fout.write('%s'%em_bboxes_t)
        '''
        '''
        fout.write(',')
        fout.write(ga_bboxes_t)
        fout.write(',')
        fout.write(mu_bboxes_t)
        fout.write(',')
        fout.write(pi_bboxes_t)
        fout.write(',')
        fout.write(pr_bboxes_t)
        '''
        #detection
        results = model.detect([image], verbose=0)
        r = results[0]
                
        '''
        print '-------------------------------'
        print 'r_class_id is '
        print r['class_ids']
        print [dataset.class_names[x]  for x in r['class_ids'] ]
        print 'r_bbox is '
        print  r['rois']
        print 'r_sores'
        print r['scores']
        '''
        reco_pdgs=[dataset.class_names[x]  for x in r['class_ids'] ]
        reco_bboxes=r['rois']
        reco_scores=r['scores']

        reco_n_em=reco_pdgs.count(11)
        reco_n_ga=reco_pdgs.count(22)
        reco_n_mu=reco_pdgs.count(13)
        reco_n_pi=reco_pdgs.count(211)
        reco_n_pr=true_pdgs.count(2212)
        
        em_scores=[]
        ga_scores=[]
        mu_scores=[]
        pi_scores=[]
        pr_scores=[]

        em_scores_mean=0.0
        ga_scores_mean=0.0
        mu_scores_mean=0.0
        pi_scores_mean=0.0
        pr_scores_mean=0.0

        em_bboxes_r=[]
        ga_bboxes_r=[]
        mu_bboxes_r=[]
        pi_bboxes_r=[]
        pr_bboxes_r=[]

        em_iou=0.0
        ga_iou=0.0
        mu_iou=0.0
        pi_iou=0.0
        pr_iou=0.0
        
        assert len(reco_pdgs)==len(reco_bboxes), 'True pdg and bbox length are NOT same!!!'

        to_remove=[]
        
        for idx in xrange(len(reco_pdgs)):
            #print idx
            pdg=reco_pdgs[idx]

            if (bb_too_small(reco_bboxes[idx])):
                to_remove.append(idx)
                if reco_pdgs[idx]==11:   reco_n_em-=1
                if reco_pdgs[idx]==22:   reco_n_ga-=1
                if reco_pdgs[idx]==13:   reco_n_mu-=1
                if reco_pdgs[idx]==211:  reco_n_pi-=1
                if reco_pdgs[idx]==2212: reco_n_pr-=1
                continue
            
            if pdg==11   :
                em_scores.append(reco_scores[idx])
                em_bboxes_r.append(reco_bboxes[idx])
                #print reco_bboxes[idx]
            if pdg==22   :
                ga_scores.append(reco_scores[idx])
                ga_bboxes_r.append(reco_bboxes[idx])
            if pdg==13   :
                mu_scores.append(reco_scores[idx])
                mu_bboxes_r.append(reco_bboxes[idx])
            if pdg==211  :
                pi_scores.append(reco_scores[idx])
                pi_bboxes_r.append(reco_bboxes[idx])
            if pdg==2212 :
                pr_scores.append(reco_scores[idx])
                pr_bboxes_r.append(reco_bboxes[idx])
        '''
        for idx in range(len(to_remove)):
            if 
            del rece_n_em[to_remove[idx]]
            del reco_n_ga[to_remove[idx]]
            del reco_n_mu[to_remove[idx]]
            del reco_n_pi[to_remove[idx]]
            del reco_n_pr[to_remove[idx]]
        '''

        if len(em_scores): em_scores_mean=sum(em_scores)/len(em_scores)
        if len(ga_scores): ga_scores_mean=sum(ga_scores)/len(ga_scores)
        if len(mu_scores): mu_scores_mean=sum(mu_scores)/len(mu_scores)
        if len(pi_scores): pi_scores_mean=sum(pi_scores)/len(pi_scores)
        if len(pr_scores): pr_scores_mean=sum(pr_scores)/len(pr_scores)
        
        if(len(em_bboxes_t) and len(em_bboxes_r)):
            em_iou=IOU(em_bboxes_t,em_bboxes_r)

        if(len(ga_bboxes_t) and len(ga_bboxes_r)):
            ga_iou=IOU(ga_bboxes_t,ga_bboxes_r)

        if(len(mu_bboxes_t) and len(mu_bboxes_r)):
            mu_iou=IOU(mu_bboxes_t,mu_bboxes_r)

        if(len(pi_bboxes_t) and len(pi_bboxes_r)):
            pi_iou=IOU(pi_bboxes_t,pi_bboxes_r)

        if(len(pr_bboxes_t) and len(pr_bboxes_r)):
            pr_iou=IOU(pr_bboxes_t,pr_bboxes_r)

            
            
            
        '''
        print r['rois']
        print [dataset.class_names[x]  for x in r['class_ids'] ]
        print r['scores']
        '''
        outdf.loc[entry]=[true_n_em,true_n_ga,true_n_mu,true_n_pi,true_n_pr,\
                          reco_n_em,reco_n_ga,reco_n_mu,reco_n_pi,reco_n_pr,\
                          em_bboxes_t,ga_bboxes_t,mu_bboxes_t,pi_bboxes_t,pr_bboxes_t,\
                          em_bboxes_r,ga_bboxes_r,mu_bboxes_r,pi_bboxes_r,pr_bboxes_r,\
                          em_scores,ga_scores,mu_scores,pi_scores,pr_scores,\
                          em_scores_mean,ga_scores_mean,mu_scores_mean,pi_scores_mean,pr_scores_mean,\
                          em_iou,ga_iou,mu_iou,pi_iou,pr_iou]
    #fout.close()
    #outdf.to_csv(test.csv)
    outdf.to_pickle('./test.pkl')
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


