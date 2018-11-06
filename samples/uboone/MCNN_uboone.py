from __future__ import absolute_import
from __future__ import division
#from __future__ import print_function

import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]="2"
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

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

#from lib.config import config_loader
#CFG=sys.argv[1]
#cfg=config_loader(CFG)

#get_ipython().magic(u'matplotlib inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


# In[2]:

particle2pdg={11:'eminus',-11:'eminus',13:'muon',-13:'muon',22:'gamma',211:'piminus',-211:'piminus',2212:'proton'}
pdg2instance={'eminus':3,'gamma':4,'muon':6,'piminus':8,'proton':9}


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
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
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

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 50

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 10
    
    DETECTION_MAX_INSTANCES = 20
    
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

        for i in range(count):
            if(verbose): sys.stdout.write("%s \n"%'>>>>load_this_entry in load_events')
            if(verbose): sys.stdout.flush()
            self.load_this_entry(i)
            #image_meta=mage.meta()
            pdgs=list([])
            bbs=list([])
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
                
                pdgs.append(roi.PdgCode())
                #print '<<<<', i, j
                #print roi.BB().size(),
                bbs.append([[roi.BB()[self.plane].tl().x, roi.BB()[self.plane].tl().y],
                            [roi.BB()[self.plane].tr().x, roi.BB()[self.plane].tr().y],
                            [roi.BB()[self.plane].bl().x, roi.BB()[self.plane].bl().y],
                            [roi.BB()[self.plane].br().x, roi.BB()[self.plane].br().y]])
                '''
                print i, roi.PdgCode()
                print roi.BB()[self.plane].tl().x,roi.BB()[self.plane].tl().y
                print roi.BB()[self.plane].tr().x,roi.BB()[self.plane].tr().y
                print roi.BB()[self.plane].bl().x,roi.BB()[self.plane].bl().y
                print roi.BB()[self.plane].br().x,roi.BB()[self.plane].br().y
                #print roi.BB().size()
                '''
            if len(pdgs):
                #image_bb=[image_meta.tl(),image_meta.tr(),image_meta.bl(),image_meta.br()]
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

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        pdgs = info['pdgs']
        bbs = info['bbs']
        count = len(pdgs)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        assert (len(bbs)==len(pdgs)), 'bbs len does not equal  '
        #print bbs
        #print pdgs
        #for bb in bbs:
            #print 'bb size ', bb
            #assert(bb.size()==3),'bb size is not 3, but is %i'%bb.size()

        #img = self.ev_instance.Image2DArray()[self.plane]
        if(verbose): sys.stdout.write("%s \n"%'>>>>load_this_entry in load_mask')
        #print '>>>>load_this_entry in load_mask'
        if(verbose): sys.stdout.flush()
        image,img_mask,_ = self.load_this_entry(image_id)
        image_meta=image.meta()
        image.binary_threshold(10,0,1)
        img_ori_np = larcv.as_ndarray(image)
        #print 'img_ori_np shape', img_ori_np.shape
        y = set(img_ori_np.flatten())
        #print y
        img_mask_np = larcv.as_ndarray(img_mask)
        #print 'img_mask_np shape', img_mask_np.shape
        #for i,pdg in enumerate(pdgs):
            #print i, pdg
        '''
        print 'image, tl, ',image_meta.tl().x,image_meta.tl().y
        print 'image, tr, ',image_meta.tr().x,image_meta.tr().y
        print 'image, bl, ',image_meta.bl().x,image_meta.bl().y
        print 'image, br, ',image_meta.br().x,image_meta.br().y
        print 'count, ',count
        '''
        for i in xrange(count):
            #if i==1 : continue
            #print i
            pdg=pdgs[i]
            #print bbs[i].size()
            bb=bbs[i]

            bb_tl=bb[0]
            bb_tr=bb[1]
            bb_bl=bb[2]
            bb_br=bb[3]
            '''
            print 'pgd,  ',pdg
            print 'bb tl, ',bb_tl[0],bb_tl[1]
            print 'bb tr, ',bb_tr[0],bb_tr[1]
            print 'bb bl, ',bb_bl[0],bb_bl[1]
            print 'bb br, ',bb_br[0],bb_br[1]
            '''
            #print image_meta.tl().x-bb.tl().x

            new_tl = (abs(image_meta.tl().x-bb_tl[0]), abs((image_meta.tl().y-bb_tl[1])/6))
            new_tr = (abs(image_meta.tl().x-bb_tr[0]), abs((image_meta.tl().y-bb_tr[1])/6))
            new_bl = (abs(image_meta.tl().x-bb_bl[0]), abs((image_meta.tl().y-bb_bl[1])/6))
            new_br = (abs(image_meta.tl().x-bb_br[0]), abs((image_meta.tl().y-bb_br[1])/6))
            
            #new_tl = (bb.tl().x-image_meta.bl().x, bb.tl().y-image_meta.bl().y))
            '''
            print 'new_tl',new_tl
            print 'new_tr',new_tr
            print 'new_bl',new_bl
            print 'new_br',new_br
            '''
            instance = pdg2instance[particle2pdg[pdg]]
            img_np_=img_mask_np.copy()
            img_np_[img_np_!=instance]=0
            img_np_[img_np_==instance]=1

            img_np_[:int(new_tl[0]), :]=0
            img_np_[int(new_tr[0]):, :]=0
            img_np_[:, int(new_bl[1]):]=0
            img_np_[:, :int(new_tl[1])]=0
            #print img_np_.shape

            img_np_=img_np_*img_ori_np
            
            img_np_=img_np_.reshape(512,512,1)
            
            mask[:,:,i:i+1]=img_np_

            
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        class_ids = np.array([self.class_names.index(s) for s in pdgs])
        return mask.copy(), class_ids

if __name__=="__main__":

    config = UbooneConfig()
    config.display()

    dataset_train = UbooneDataset("/data/dayajun/toymodel/uboone/train_data/75_200.root")
    dataset_train.load_events(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_train.prepare()
    
    dataset_val = UbooneDataset("/data/dayajun/toymodel/uboone/train_data/75_200_val.root")
    dataset_val.load_events(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
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
        print "...............Start Training II"
        model_path="/data/dayajun/sw/Mask_RCNN/logs/particles20181001T1620/mask_rcnn_particles_0100.h5"
        model.load_weights(model_path, by_name=True)
        model.train(dataset_train, dataset_val, 
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=200, 
                    layers="all")

    if sys.argv[1]=='heads_all':
        print "...............Start Training I"
        #model_path="/data/dayajun/sw/Mask_RCNN/logs/particles20180927T1747/mask_rcnn_particles_0050.h5"
        #model_path="/data/dayajun/sw/Mask_RCNN/logs/particles20181010T2123/mask_rcnn_particles_0028.h5"
        model_path=""
        if model_path:
            model.load_weights(model_path, by_name=True)
        #'''
        model.train(dataset_train, dataset_val, 
                    learning_rate=config.LEARNING_RATE, 
                    epochs=100, 
                    layers='heads')
        #'''
        print "...............Start Training II"
        model.train(dataset_train, dataset_val, 
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=200, 
                    layers="all")
        
    print "Training Done!"

