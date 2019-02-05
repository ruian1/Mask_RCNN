from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="5,6,7,8"
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Force print 

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

if (len(sys.argv)>3):
    path_train=sys.argv[2]
    path_val=sys.argv[3]
    path_model=""
if len(sys.argv) == 5:
    path_model=sys.argv[4]

#from lib.config import config_loader
#CFG=sys.argv[1]
#cfg=config_loader(CFG)

#get_ipython().magic(u'matplotlib inline')

# Directory to save logs and trained model
#MODEL_DIR = os.path.join(ROOT_DIR, "logs")
MODEL_DIR = os.path.join("/scratch/ruian/maskrcnn_log")

# In[2]:

pdg2particle={0:'BG', 11:'eminus',-11:'eminus',13:'muon',-13:'muon',22:'gamma',211:'piminus',-211:'piminus',2212:'proton'}
#particle2instance={'eminus':3,'gamma':4,'muon':6,'piminus':8,'proton':9}
particle2instance={'BG':0.1,'eminus':3,'gamma':3,'muon':6,'piminus':8,'proton':9}


# In[3]:
verbose=0

from larcv import larcv


# In[4]:

class UbooneConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Particles"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 4
    IMAGES_PER_GPU = 5
    
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 8  # background + particles

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    #RPN_ANCHOR_RATIOS = [0.125,0.5,  1,  2,  4]
    RPN_ANCHOR_RATIOS = [0.5,1,2]
    RPN_ANCHOR_SCALES = (8,  16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple,100 in the shape.ipynb sample
    STEPS_PER_EPOCH = 200

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 80
    
    DETECTION_MAX_INSTANCES = 15
    
    USE_MINI_MASK = False
    #MINI_MASK_SHAPE = (128, 128)  # (height, width) of the mini-mask

    IMAGE_CHANNEL_COUNT =1
    
    IMAGE_RESIZE_MODE = 'none'

    Train_BN=False

    MEAN_PIXEL = np.array([0.001])

def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def getRGBfromI(RGBint):
    #print RGBint
    RGBint=int(RGBint)
    blue =  RGBint & 255
    green = (RGBint >> 8) & 255
    red =   (RGBint >> 16) & 255
    return red, green, blue

def xy2bb(xy, this_bl):

    tl=[np.min(xy[:,1]), np.max(xy[:,0])]
    tr=[np.max(xy[:,1]), np.max(xy[:,0])]
    bl=[np.min(xy[:,1]), np.min(xy[:,0])]
    br=[np.max(xy[:,1]), np.min(xy[:,0])]

    new_tl=[tl[0]+this_bl[0], tl[1]*6+this_bl[1]]
    new_tr=[tr[0]+this_bl[0], tr[1]*6+this_bl[1]]
    new_bl=[bl[0]+this_bl[0], bl[1]*6+this_bl[1]]
    new_br=[br[0]+this_bl[0], br[1]*6+this_bl[1]]

    return [new_tl,new_tr,new_bl,new_br]

class UbooneDataset(utils.Dataset):
    def __init__(self,input_file):
        self.iom=larcv.IOManager(0) 
        self.iom.add_in_file(input_file)
        self.iom.initialize()
        self.counter=1
        super(UbooneDataset, self).__init__()
    
    def load_events(self, count, height, width):
        # Add classes
        self.plane=2

        self.add_class("Particles", 1, 11)
        self.add_class("Particles", 2, -11)
        self.add_class("Particles", 3, 13)
        self.add_class("Particles", 4, -13)
        self.add_class("Particles", 5, 22)
        self.add_class("Particles", 6, 211)
        self.add_class("Particles", 7, -211)
        self.add_class("Particles", 8, 2212)
        '''
        self.add_class("Particles", 1, 11)
        self.add_class("Particles", 2, 13)
        self.add_class("Particles", 3, 22)
        self.add_class("Particles", 4, 211)
        self.add_class("Particles", 5, 2212)
        '''
        for i in range(count):
            if(verbose): sys.stdout.write("%s \n"%'>>>>load_this_entry in load_events')
            if(verbose): sys.stdout.flush()
            self.load_this_entry(i)
            #image_meta=mage.meta()
            pdgs=list([])
            bbs=list([])

            #print ('%ith event has %i pars,'%(i,self.ev_roi.ROIArray().size()))
            
            for j, roi in enumerate(self.ev_roi.ROIArray()):
                #if j==0 : continue # First ROI name null with producer of iseg
                if roi.PdgCode()==16: continue #nu_tau
                if roi.PdgCode()==111: continue #pi_zero...
                if roi.PdgCode()==321: continue #Kplus...
                if roi.PdgCode()==2112: continue #Delta...
                if roi.PdgCode()==1000010020: continue #Dutron...
                if roi.PdgCode()==1000010030: continue #Tritium...
                if roi.PdgCode()==1000010040: continue #Alpha...   
                if roi.PdgCode()==1000020030: continue #Alpha...                
                if roi.PdgCode()==1000020040: continue #Alpha... 

                #print ('%ith par is %i'%(j, roi.PdgCode()))
                
                if ((roi.PdgCode()==22 or roi.PdgCode()==11 or roi.PdgCode()==-11) and True):
                    flag_dbscan_applied=True
                    image=self.ev_image.Image2DArray()[2]#using plane 2 only
                    meta = image.meta()
                    image.threshold(10,0)
                    image_np=larcv.as_ndarray(image)
                    X = np.argwhere(image_np>0)

                    if (0 not in X.shape):
                        db = DBSCAN(eps=12, min_samples=15).fit(X)
                        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                        core_samples_mask[db.core_sample_indices_] = True
                        labels = db.labels_
                        unique_labels = set(labels)
                        for k in unique_labels:
                            if k == -1: continue
                            class_member_mask = (labels == k)
                            xy = X[class_member_mask & core_samples_mask]
                            this_bl=[meta.bl().x, meta.bl().y]

                            bb_this=xy2bb(xy, this_bl)

                            new_tl=bb_this[0]
                            new_br=bb_this[3]
                            
                            if(abs(new_tl[0]-new_br[0])<10 and abs(new_tl[1]-new_br[1])/6.<10):
                                continue
                            bb_old = [[roi.BB()[self.plane].tl().x, roi.BB()[self.plane].tl().y],
                                      [roi.BB()[self.plane].tr().x, roi.BB()[self.plane].tr().y],
                                      [roi.BB()[self.plane].bl().x, roi.BB()[self.plane].bl().y],
                                      [roi.BB()[self.plane].br().x, roi.BB()[self.plane].br().y]]

                            pdgs.append(11)
                            bbs.append(bb_this)
                
                else:
                    #if(abs(roi.BB()[self.plane].tl().x-roi.BB()[self.plane].tr().x)>10 and
                    #   abs(roi.BB()[self.plane].tl().x-roi.BB()[self.plane].bl().y)/6.>10):
                    if (True):
                        pdgs.append(abs(roi.PdgCode()))
                        bbs.append([[roi.BB()[self.plane].tl().x, roi.BB()[self.plane].tl().y],
                                    [roi.BB()[self.plane].tr().x, roi.BB()[self.plane].tr().y],
                                    [roi.BB()[self.plane].bl().x, roi.BB()[self.plane].bl().y],
                                    [roi.BB()[self.plane].br().x, roi.BB()[self.plane].br().y]])
            if len(pdgs):
                #image_bb=[image_meta.tl(),image_meta.tr(),image_meta.bl(),image_meta.br()]
                self.add_image("Particles", image_id=i, path=None,
                               width=width, height=height,
                               bg_color=0, pdgs=pdgs, bbs=bbs)#,image_bb=image_bb)

            else:#otherwise image_id and root file idx are mismatched
                pdgs.append(0)
                bbs.append([[roi.BB()[self.plane].tl().x, roi.BB()[self.plane].tl().y],
                            [roi.BB()[self.plane].tr().x, roi.BB()[self.plane].tr().y],
                            [roi.BB()[self.plane].bl().x, roi.BB()[self.plane].bl().y],
                            [roi.BB()[self.plane].br().x, roi.BB()[self.plane].br().y]])
                self.add_image("Particles", image_id=i, path=None,
                               width=width, height=height,
                               bg_color=0, pdgs=pdgs, bbs=bbs)#,image_bb=image_bb)

            #for x in xrange(len(self.image_info)):
            
            #print '>>>>>', x
            #for y in xrange(len(self.image_info[x]['bbs'])):
            #    if not self.image_info[x]['bbs'][y].size() ==3:
            #        print '@image id ', x,  self.image_info[x]['bbs'][y].size()

    def load_this_entry(self, entry):
        if(verbose): sys.stdout.write("%s, %i \n"%('>>>>counter', self.counter))
        #print '>>>>counter>>',self.counter
        self.counter+=1
        self.iom.read_entry(entry, True)
        self.ev_image    = self.iom.get_data(larcv.kProductImage2D,"wire")
        self.ev_roi      = self.iom.get_data(larcv.kProductROI,"rui")
        self.ev_instance = self.iom.get_data(larcv.kProductImage2D,"segment_rui")
        #print "run", self.ev_image.run(),", subrun", self.ev_image.subrun(), ", event",self.ev_image.event()
        if(verbose): sys.stdout.write("run %s, subrun %s, event %s \n"%(self.ev_image.run(),self.ev_image.subrun(),self.ev_image.event()))
        if(verbose): sys.stdout.flush()
        #print ev_image.Image2DArray().size()
        return self.ev_image.Image2DArray()[self.plane], self.ev_instance.Image2DArray()[self.plane],self.ev_roi.ROIArray()

    
    def load_image(self, image_id):
        if(verbose): sys.stdout.write("%s \n"%'>>>>load_this_entry in load_image')
        #print '>>>>load_this_entry in load_image'
        if(verbose): sys.stdout.flush()
        image,_,_ = self.load_this_entry(image_id)
        img_np_=larcv.as_ndarray(image)
        #print 'before thresholding, sum is ', np.sum(img_np_)
        #print img_np_[100,:]
        image.threshold(10,0) #thershold value here 
        img_np=larcv.as_ndarray(image)
        #print 'after thresholding, sum is ', np.sum(img_np_)
        #print image_np[100,:]
        img_np=img_np.reshape(512,512,1)
        return img_np.copy()
        #img_np3=np.repeat(img_np,3).reshape(512,512,3)
        #img_np3=np.round(img_np3,0)
        #return img_np3

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "Particles":
            return info["Particles"]
        else:
            super(self.__class__).image_reference(self, image_id)


    def load_bbox(self, image_id):
        info = self.image_info[image_id]
        pdgs = info['pdgs']
        bbs = info['bbs']
        count = len(pdgs)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        assert (len(bbs)==len(pdgs)), 'bbs len does not equal  '

        image,img_mask,_ = self.load_this_entry(image_id)
        image_meta=image.meta()
        image.binary_threshold(10,0,1)
        img_ori_np = larcv.as_ndarray(image)

        y = set(img_ori_np.flatten())

        img_mask_np = larcv.as_ndarray(img_mask)

        boxes=np.zeros((count,4,2))

        for i in xrange(count):
            pdg=pdgs[i]
            bb=bbs[i]

            bb_tl=bb[0]
            bb_tr=bb[1]
            bb_bl=bb[2]
            bb_br=bb[3]

            new_tl = np.array([abs(image_meta.tl().x-bb_tl[0]), abs((image_meta.tl().y-bb_tl[1])/6)])
            new_tr = np.array([abs(image_meta.tl().x-bb_tr[0]), abs((image_meta.tl().y-bb_tr[1])/6)])
            new_bl = np.array([abs(image_meta.tl().x-bb_bl[0]), abs((image_meta.tl().y-bb_bl[1])/6)])
            new_br = np.array([abs(image_meta.tl().x-bb_br[0]), abs((image_meta.tl().y-bb_br[1])/6)])
            
            #if pdg==11: #this is introduced by some bug when generating bbox
            new_tl[1]= 512-new_tl[1]
            new_tr[1]= 512-new_tr[1]
            new_bl[1]= 512-new_bl[1]
            new_br[1]= 512-new_br[1]

            boxes[i]=np.array([new_tl,new_tr,new_bl,new_br])

        return boxes

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        pdgs = info['pdgs']
        bbs = info['bbs']
        count = len(pdgs)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        assert (len(bbs)==len(pdgs)), 'bbs len does not equal  '

        #img = self.ev_instance.Image2DArray()[self.plane]
        if(verbose): sys.stdout.write("%s \n"%'>>>>load_this_entry in load_mask')
        #print '>>>>load_this_entry in load_mask'
        if(verbose): sys.stdout.flush()
        image,img_mask,_ = self.load_this_entry(image_id)
        image_meta=image.meta()
        image.binary_threshold(10,0,1)
        img_ori_np = larcv.as_ndarray(image)

        #y = set(img_ori_np.flatten())
        
        img_mask_np = larcv.as_ndarray(img_mask)

        instance_set=set(img_mask_np.flatten())

        #print ('instance set is, ', instance_set)
        
        for i in xrange(count):
            #if i==1 : continue
            #print i
            pdg=pdgs[i]
            
            bb=bbs[i]

            bb_tl=bb[0]
            bb_tr=bb[1]
            bb_bl=bb[2]
            bb_br=bb[3]

            #print pdg
            
            if pdg==11: #this is introduced by some bug when generating bbox
                new_tl = np.array([abs(image_meta.tl().x-bb_tl[0]), abs((image_meta.tl().y-bb_tl[1])/6)])
                new_tr = np.array([abs(image_meta.tl().x-bb_tr[0]), abs((image_meta.tl().y-bb_tr[1])/6)])
                new_bl = np.array([abs(image_meta.tl().x-bb_bl[0]), abs((image_meta.tl().y-bb_bl[1])/6)])
                new_br = np.array([abs(image_meta.tl().x-bb_br[0]), abs((image_meta.tl().y-bb_br[1])/6)])
                        
                new_tl[1]= 512-new_tl[1]
                new_tr[1]= 512-new_tr[1]
                new_bl[1]= 512-new_bl[1]
                new_br[1]= 512-new_br[1]
                
                instance = particle2instance[pdg2particle[pdg]]
                img_np_=img_mask_np.copy()

                np_eminus=img_np_==3
                np_gamma =img_np_==4
                img_np_=np_eminus+np_gamma

                xl=int(min(new_tl[0],new_tr[0]))
                xr=int(max(new_tl[0],new_tr[0]))
                img_np_[:, :xl]=0
                img_np_[:, xr:]=0
                
                yb=int(min(new_bl[1],new_tl[1]))
                yt=int(max(new_bl[1],new_tl[1]))
                img_np_[:yb, :]=0
                img_np_[yt:, :]=0
                img_np_=img_np_*img_ori_np
                img_np_=img_np_.reshape(512,512,1)
                mask[:,:,i:i+1]=img_np_
            else:        

                new_tl = (max(bb_tl[0]-image_meta.tl().x, 0), abs(512-(bb_tl[1]-image_meta.bl().y)/6))
                new_tr = (max(bb_tr[0]-image_meta.tl().x, 0), abs(512-(bb_tr[1]-image_meta.bl().y)/6))
                new_bl = (max(bb_bl[0]-image_meta.tl().x, 0), abs(512-(bb_bl[1]-image_meta.bl().y)/6))
                new_br = (max(bb_br[0]-image_meta.tl().x, 0), abs(512-(bb_br[1]-image_meta.bl().y)/6))

                #print new_tl
                #print new_tr
                #print new_bl
                #print new_br
                
                instance = particle2instance[pdg2particle[pdg]]
                img_np_=img_mask_np.copy()
                
                np_eminus=img_np_==3
                np_gamma =img_np_==4

                #temporary fix for p- and p+, 

                if pdg==211:
                    np_muon=img_np_==6
                    '''
                    np_proton=img_np_==9
                    img_np_[img_np_==instance]=1
                    img_np_+=np_proton
                    img_np_proton_from_pion=img_np_.copy()
                    img_np_proton_from_pion[:int(new_tl[0]), :]=0
                    img_np_proton_from_pion[int(new_tr[0]):, :]=0
                    img_np_proton_from_pion[:, int(new_bl[1]):]=0
                    img_np_proton_from_pion[:, :int(new_tl[1])]=0
                    if np.sum(img_np_proton_from_pion) > 0:
                        img_np_proton_from_pion=img_np_proton_from_pion*img_ori_np
                        img_np_proton_from_pion=img_np_proton_from_pion.reshape(512,512,1)
                        np.concatenate((mask, img_np_proton_from_pion),axis=2)
                        pdgs.append(2212)
                        newbox= (utils.extract_bboxes(img_np_proton_from_pion)[0])
                        lx=newbox[0]
                        hx=newbox[2]
                        ly=newbox[1]
                        hy=newbox[3]
                        bb_proton_from_pion = [[lx,ly],
                                               [hx,ly],
                                               [lx,hy],
                                               [hx,hy]]
                        bbs.append(bb_proton_from_pion)
                    '''
                #print ('instance is ', instance)
                img_np_[img_np_!=instance]=0
                img_np_[img_np_==instance]=1
                img_np_+=np_eminus
                img_np_+=np_gamma
                if pdg==211:
                    img_np_+=np_muon
                    #print "wtf?"
                    #print new_tl
                    #print new_tr
                    #print new_bl
                    #print new_br
                img_np_[:int(new_tl[0]), :]=0
                img_np_[int(new_tr[0]):, :]=0
                img_np_[:, int(new_bl[1]):]=0
                img_np_[:, :int(new_tl[1])]=0

                img_np_=img_np_*img_ori_np
                img_np_=img_np_.reshape(512,512,1)
                mask[:,:,i:i+1]=img_np_

        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)

        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        class_ids = np.array([self.class_names.index(s) for s in pdgs])

        res_count=0
        use_mask_idx=[]
        for i in range(mask.shape[-1]):
            if (np.sum(mask[:,:,i])==0):continue
            res_count+=1
            use_mask_idx.append(i)

        res_mask = np.zeros([info['height'], info['width'], res_count], dtype=np.uint8)
        
        for i in range(len(use_mask_idx)):
            res_mask[:,:,i]=mask[:,:,use_mask_idx[i]]

        res_class_ids=np.array([class_ids[use_mask_idx[i]] for i in range(len(use_mask_idx))])
        
        #return mask.copy(), class_ids
        return res_mask, res_class_ids

if __name__=="__main__":

    config = UbooneConfig()
    config.display()

    dataset_train = UbooneDataset(path_train)
    dataset_train.load_events(9990, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    #dataset_train.load_events(100, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_train.prepare()
    
    dataset_val = UbooneDataset(path_val)
    dataset_val.load_events(1990, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    #dataset_val.load_events(100, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_val.prepare()
    
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)
    """
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["conv1","mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
    """

    if sys.argv[1]=='test':
        class InferenceConfig(UbooneConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        inference_config = InferenceConfig()

        image_id=2
        print 'image_id ',image_id
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_train, inference_config, 
                                   image_id, use_mini_mask=False)

        print np.sum(original_image)
    

    if sys.argv[1]=='heads':
        print "...............Start Training I"
        model.train(dataset_train, dataset_val, 
                    learning_rate=config.LEARNING_RATE, 
                    epochs=10, 
                    layers='heads')

    if sys.argv[1]=='all':
        print "...............Start Training I"
        model_path=path_model
        model.load_weights(model_path, by_name=True)
        '''
        model.train(dataset_train, dataset_val, 
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=100, 
                    layers="all")
        print "...............Start Training I"
        '''
        model.train(dataset_train, dataset_val, 
                    learning_rate=config.LEARNING_RATE,
                    epochs=200, 
                    layers="all")

    if sys.argv[1]=='heads_all':
        print "...............Start Training I"
        model_path=path_model
        if model_path:
            model.load_weights(model_path, by_name=True)
        #'''
        model.train(dataset_train, dataset_val, 
                    learning_rate=config.LEARNING_RATE / 100, 
                    epochs=100, 
                    layers='heads')
        #'''
        print "...............Start Training II"
        model.train(dataset_train, dataset_val, 
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=100, 
                    layers="all")
        
    print "Training Done!"

