import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "Particles"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 5
    
    BACKBONE_STRIDES = [4, 8, 16, 32, 64, 128, 256]
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 8  # background + particles

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_RATIOS = [0.5,  1,  2,  4,   8,  16, 32]
    RPN_ANCHOR_SCALES = (8  , 16, 32, 64, 128, 256, 512)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 10
    
    DETECTION_MAX_INSTANCES = 20
    
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (128, 128)  # (height, width) of the mini-mask

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

class ShapesDataset(utils.Dataset):
    def __init__(self,input_file):
        self.iom=larcv.IOManager(0) 
        self.iom.add_in_file(input_file)
        self.iom.initialize()
        self.counter=1
        super(ShapesDataset, self).__init__()
    
    def load_shapes(self, count, height, width):
        # Add classes
        self.add_class("Particles", 1, 11)
        self.add_class("Particles", 2, -11)
        self.add_class("Particles", 3, 13)
        self.add_class("Particles", 4, -13)
        self.add_class("Particles", 5, 22)
        self.add_class("Particles", 6, 211)
        self.add_class("Particles", 7, -211)
        self.add_class("Particles", 8, 2212)

        for i in range(count):
            if(verbose): sys.stdout.write("%s \n"%'>>>>load_this_entry in load_shapes')
            if(verbose): sys.stdout.flush()
            self.load_this_entry(i)
            pdgs=[]
                
            for j, roi in enumerate(self.ev_roi.ROIArray()):
                if j==0 : continue # First ROI name null with producer of iseg
                if roi.PdgCode()==111: continue #pi_zero...
                if roi.PdgCode()==321: continue #Kplus...
                if roi.PdgCode()==2112: continue #Delta...
                if roi.PdgCode()==1000010020: continue #Dutron...
                if roi.PdgCode()==1000010030: continue #Tritium...
                if roi.PdgCode()==1000010040: continue #Alpha...   
                if roi.PdgCode()==1000020030: continue #Alpha...                
                if roi.PdgCode()==1000020040: continue #Alpha... 


                pdgs.append(roi.PdgCode())
            self.add_image("Particles", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=0, pdgs=pdgs)
    
    def load_this_entry(self, entry):
        if(verbose): sys.stdout.write("%s, %i \n"%('>>>>counter', self.counter))
        #print '>>>>counter>>',self.counter
        self.counter+=1
        self.iom.read_entry(entry)
        self.ev_image    = self.iom.get_data(larcv.kProductImage2D,"wire")
        self.ev_roi      = self.iom.get_data(larcv.kProductROI,"iseg")
        self.ev_instance = self.iom.get_data(larcv.kProductImage2D,"segment")
        self.plane=2
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
        image.threshold(10,0) #thershold value here 
        img_np=larcv.as_ndarray(image)
        #print 'after thresholding, sum is ', np.sum(img_np_)
        img_np=img_np.reshape(512,512,1)
        img_np3=np.repeat(img_np,3).reshape(512,512,3)
        img_np3=np.round(img_np3,0)
        return img_np3

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "Particles":
            return info["Particles"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        pdgs = info['pdgs']
        count = len(pdgs)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)

        #img = self.ev_instance.Image2DArray()[self.plane]
        if(verbose): sys.stdout.write("%s \n"%'>>>>load_this_entry in load_mask')
        #print '>>>>load_this_entry in load_mask'
        if(verbose): sys.stdout.flush()
        image,img_mask,_ = self.load_this_entry(image_id)
        image.binary_threshold(0,0,1)
        img_ori_np = larcv.as_ndarray(image)
        #print 'img_ori_np shape', img_ori_np.shape
        y = set(img_ori_np.flatten())
        #print y
        img_mask_np = larcv.as_ndarray(img_mask)
        #print 'img_mask_np shape', img_mask_np.shape
        for i,pdg in enumerate(pdgs):
            instance = pdg2instance[particle2pdg[pdg]]
            img_np_=img_mask_np.copy()
            img_np_[img_np_!=instance]=0
            img_np_[img_np_==instance]=1

            img_np_=img_np_*img_ori_np
            
            img_np_=img_np_.reshape(512,512,1)
            
            mask[:,:,i:i+1]=img_np_
            
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        class_ids = np.array([self.class_names.index(s) for s in pdgs])
        return mask, class_ids

if __name__=="__main__":

    config = ShapesConfig()
    config.display()

    dataset_train = ShapesDataset("/data/dayajun/toymodel/uboone/train_data/75_200.root")
    dataset_train.load_shapes(100, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_train.prepare()
    
    dataset_val = ShapesDataset("/data/dayajun/toymodel/uboone/train_data/75_200_val.root")
    dataset_val.load_shapes(20, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_val.prepare()
    
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    print sys.argv[1]
    print sys.argv[1]=='heads'
    if sys.argv[1]=='heads':
        print "...............Start Training I"
    
        model.train(dataset_train, dataset_val, 
                    learning_rate=config.LEARNING_RATE, 
                    epochs=20, 
                    layers='heads')

    if sys.argv[1]=='all':
        print "...............Start Training II"
        model_path="/data/dayajun/sw/Mask_RCNN/logs/particles20180920T1657/mask_rcnn_particles_0020.h5"
        model.load_weights(model_path, by_name=True)
        model.train(dataset_train, dataset_val, 
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=30, 
                    layers="all")

    print "Training Done!"

