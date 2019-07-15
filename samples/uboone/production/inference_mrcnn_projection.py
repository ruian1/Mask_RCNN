import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
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
from lib import utility as u
from lib import bbox_helper as bh
from lib import contour_helper as ch
from lib import mrcnn_result_analyze as ma
from lib import line_helper as lh
from ROOT import TChain

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

from MCNN_uboone_updated_debug import UbooneConfig

class InferenceConfig(UbooneConfig):
    #Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    #DETECTION_MAX_INSTANCES=10
    #RPN_ANCHOR_RATIOS = [0.125,0.5,1,2,4]
    DETECTION_MIN_CONFIDENCE = 0.6

def main(IMAGE_FILE, VTX_FILE, TRACK_FILE, OUT_DIR, CFG):

    # 
    # Read VTX and reco_Track ASS
    #
    # Vertex TChain
    vertex_chain = TChain("vertex_trackReco_tree")
    vertex_chain.AddFile(TRACK_FILE)
    # Track TChain
    track_chain = TChain("track_trackReco_tree")
    track_chain.AddFile(TRACK_FILE)
    # Ass TChain
    ass_chain = TChain("ass_trackReco_tree")
    ass_chain.AddFile(TRACK_FILE)
    
    class_names=[0, 11, 22, 13, 211, 2212, 2213]

    #
    # Loading configure
    #
    cfg  = config_loader(CFG)
    assert cfg.batch == 1

    #
    # Construct output tree
    #
    rd = ROOTData()
    NUM = int(os.path.basename(VTX_FILE).split(".")[0].split("_")[-1])
    FOUT = os.path.join(OUT_DIR,"maskrcnn_out_%04d.root" % NUM)
    #FOUT = os.path.join(OUT_DIR,"multimaskrcnn_out_04.root")
    tfile = ROOT.TFile.Open(FOUT,"RECREATE")
    tfile.cd()
    tree  = ROOT.TTree("maskrcnn_tree","")
    rd.init_tree(tree)
    rd.reset()

    #
    #>>>>>>>>> Loading Network... <<<<<<<<<<#
    #
    import MCNN_uboone_updated_debug
    config = UbooneConfig()
    #config.display()
    config = InferenceConfig()
    MODEL_DIR=OUT_DIR

    model = modellib.MaskRCNN(mode="inference", model_dir=OUT_DIR,config=config)
    weight_file = ""
    exec("weight_file = cfg.weight_file_mrcnn_plane2")
    model.load_weights(weight_file, by_name=True)

    #oiom = larcv.IOManager(larcv.IOManager.kWRITE)
    #oiom.set_out_file("trash.root")
    #oiom.initialize()

    #
    #initialize iomanager
    #

    iom  = larcv.IOManager(larcv.IOManager.kREAD)
    iom.add_in_file(IMAGE_FILE)
    iom.add_in_file(VTX_FILE)
    iom.initialize()


    for entry in xrange(iom.get_n_entries()):
    #entry = 0
    #while (entry < iom.get_n_entries()):
     
        iom.read_entry(entry)

        ev_pgr = iom.get_data(larcv.kProductPGraph,"inter_par")
        ev_par = iom.get_data(larcv.kProductPixel2D,"inter_par_pixel")
        ev_pix = iom.get_data(larcv.kProductPixel2D,"inter_img_pixel")
        ev_int = iom.get_data(larcv.kProductPixel2D,"inter_int_pixel")
        ev_img = iom.get_data(larcv.kProductImage2D,"wire")
        
        print '=====>>>', entry
        print 'working on run, subrun, event',ev_pix.run(),ev_pix.subrun(),ev_pix.event()

        rd.run[0]    = int(ev_pix.run())
        rd.subrun[0] = int(ev_pix.subrun())
        rd.event[0]  = int(ev_pix.event())
        rd.entry[0]  = int(iom.current_entry())

        rd.num_vertex[0] = int(ev_pgr.PGraphArray().size())

        # Read in vertex and track Ass if there is an vertex.
        vertex_chain.GetEntry(entry)
        vertex_v = vertex_chain.vertex_trackReco_branch
        track_chain.GetEntry(entry)
        track_v = track_chain.track_trackReco_branch
        ass_chain.GetEntry(entry)
        ass_v = ass_chain.ass_trackReco_branch
        # Skip if there is no reconstructed tracks
        if not ass_v.size():
            print "Skipping...No reconstructed tracks..."
            continue
        vertex_track_ass = ass_v.association(vertex_v.id(), track_v.id())
        assert ev_img.run()==ass_v.run()
        assert ev_img.subrun()==ass_v.subrun()
        assert ev_img.event()==ass_v.event_id()
        print "vertex_track_ass has %i associations.", vertex_track_ass.size()
        print "ass_v has %i associations.", ass_v.size()
                
        for ix,pgraph in enumerate(ev_pgr.PGraphArray()):
            print "@ %dth vertex..." % ix

            rd.vtxid[0] = int(ix)
   
            pgr = ev_pgr.PGraphArray().at(ix)
            cindex_v = np.array(pgr.ClusterIndexArray())
            
            #pixel2d_par_vv = ev_par.Pixel2DClusterArray()
            pixel2d_pix_vv = ev_pix.Pixel2DClusterArray()
            pixel2d_int_vv = ev_int.Pixel2DClusterArray()

            x,y,z=-1,-1,-1
            if (pgraph.ParticleArray().size()) :
                roi0 = pgraph.ParticleArray().front()
                x,y,z = roi0.X(), roi0.Y(), roi0.Z()
                       
            for plane in xrange(3):
                if plane!=2 : continue
                if (pgraph.ParticleArray().size()) :

                    x_2d, y_2d = ROOT.Double(), ROOT.Double()
                    whole_img = ev_img.at(plane)
                    meta=pgraph.ParticleArray().front().BB()
                    
                    x_2d, y_2d = u.Project3Dto2D_and_fliplr(whole_img.meta(), x, y, z, plane, x_2d, y_2d)
                    new_y_2d, new_x_2d = u.Meta_origin_helper(x_2d, y_2d, get_new_origin=1)
                    vtx_x_2d, vtx_y_2d = x_2d, y_2d
                    
                    meta_crop = larcv.ImageMeta(512,512*6,512,512,0,8448,plane)
                    meta_origin_x, meta_origin_y = u.Meta_origin_helper(x_2d, y_2d, verbose=0)
                    meta_crop.reset_origin(meta_origin_x, meta_origin_y)
                    
                    vertex_image = ev_img.at(plane).crop(meta_crop)
                    vertex_image = larcv.as_ndarray(vertex_image)
                    vertex_image_modified=u.Nparray_modify(vertex_image)
                    vertex_image_grey = np.where(vertex_image > 0, 1, 0).astype(np.uint8)
                    linesP=lh.HoughLinesP(vertex_image_grey.copy())
                    #print "nan sum after 333333333 ", np.sum(np.isnan(vertex_image))
                    
                    # sometime houghlinesP changes the vertex image,
                    # I don't know why. So just keep running till
                    # one works...

                    if np.sum(np.isnan(vertex_image)) :
                        print "OOM!! skipping entry%i @%ix........"%(entry,ix)
                        continue
                    
                    vertex_image_copy = vertex_image.copy()
                    #vertex_image_grey_copy = vertex_image_grey.copy()

                    if linesP is not None:
                        for i in range(0, len(linesP)):
                            # idx = 6
                            # for i in range(idx, idx+1):
                            line_i = linesP[i][0]
                            #ax2.plot([line_i[0], line_i[2]], [line_i[1], line_i[3]], lw=0.1, alpha=1)
                            # Check if line seg is close to image edge
                            if not lh.Line_close_to_edge(line_i):
                                continue
                            connected_lines=[]
                            # print "line %i find these,"%i
                            for each in lh.Lines_connections(i, linesP, connected_lines):
                                line_each = linesP[each][0]
                                #ax2.plot([line_each[0], line_each[2]], [line_each[1], line_each[3]], lw=4, alpha=1)
                                pt_start=(line_each[0], line_each[1])
                                pt_end=(line_each[2], line_each[3])
                                cv2.line(vertex_image, pt_start, pt_end, 0, 5)
                                #cv2.line(vertex_image_grey_copy, pt_start, pt_end, 1000, 3)

                    print "making cosmic-removed np image ..."
                    vertex_image_nocosmic_modified=u.Nparray_modify(vertex_image)
                    
                    '''
                    fig,ax=plt.subplots(1,1,figsize=(8,6))
                    print vertex_image_modified.shape
                    #ax.imshow(vertex_image_modified.reshape(cfg.xdim, cfg.ydim))
                    fig.savefig("%i_%i_%i_%i.pdf"%(ev_pix.run(),ev_pix.subrun(),ev_pix.event(),ix), bbox_inches='tight')
                    '''

                    # Read in reco track and project to 2d
                    projected_tracks = {}            
                    for track_idx in vertex_track_ass[ix]:
                        each_track = track_v[track_idx]
                        x_=[]
                        y_=[]
                        for idx in xrange(each_track.NumberTrajectoryPoints()):
                            x = each_track.LocationAtPoint(idx).x()
                            y = each_track.LocationAtPoint(idx).y()
                            z = each_track.LocationAtPoint(idx).z()
                            x_2d = ROOT.Double()
                            y_2d = ROOT.Double()
                            x_2d, y_2d = u.Project3Dto2D_and_fliplr(whole_img.meta(), x, y, z, plane, x_2d, y_2d)
                            x_2d =  x_2d - vtx_x_2d + new_x_2d
                            y_2d =  y_2d - vtx_y_2d + new_y_2d
                            
                            #                     new_y_2d, new_x_2d = meta_2dpt_helper(x_2d, y_2d, get_new_origin=1)
                            #                     ax2.plot(new_y_2d, new_x_2d, "*", markersize=1, color="black")
                            
                            x_.append(y_2d)
                            y_.append(x_2d)
                            
                        projected_track = np.zeros((len(x_), 2))
                        projected_track[:,0]=x_
                        projected_track[:,1]=y_
                        projected_tracks[np.int(track_idx)] = projected_track
                    '''
                    fig,(ax0, ax1, ax2)=plt.subplots(1,3,figsize=(21,7))
                    ax2.set_xlim(0, 512)
                    ax2.set_ylim(0, 512)
                    '''
                    projected_track_contours = {}
                    
                    for track_id, track in projected_tracks.iteritems():
                        #ax2.plot(track[:,0], track[:,1], "*", markersize=1)
                        track_contour = np.array(track).reshape((-1,1,2)).astype(np.int32)
                        vertex_image_zeros = np.zeros_like(vertex_image)
                        cv2.drawContours(vertex_image_zeros,[track_contour],0, 2 ,2)
                        img, contours, hierarchy = u.Find_contours(vertex_image_zeros)

                        projected_track_contours[np.int(track_id)] = contours
                        
                        for contour in contours:
                            verts = contour[:,0,:]
                            #ax2.plot(verts[:,0], verts[:,1], "*", markersize=2)

                    #Detection-1, for image with vectex centered
                    #from datetime import datetime
                    #a = datetime.now()

                    print "Dectecting..."
                    results_center = model.detect([vertex_image_modified], verbose=0)
                    results_center_nocosmic = model.detect([vertex_image_nocosmic_modified], verbose=0)
                                                            
                    #b = datetime.now()
                    #c=b-a
                    #print 'using time of %i seconds'%c.seconds
                
                    r_center = results_center[0]
                    r_center_nocosmic = results_center_nocosmic[0]

                    ma.Vertex_based_analyze(rd, "center", r_center, vtx_x_2d, vtx_y_2d)
                    ma.Mask_based_analyze(  rd, "center", r_center, projected_track_contours)

                    rd.inferred[0] = 1
                    '''
                    visualize.display_instances(vertex_image_modified, r_center['rois'],
                                                r_center['masks'], r_center['class_ids'],
                                                class_names, r_center['scores'], ax=ax0,
                                                title="center_Predictions")

                    visualize.display_instances(vertex_image_nocosmic_modified, r_center_nocosmic['rois'],
                                                r_center_nocosmic['masks'], r_center_nocosmic['class_ids'],
                                                class_names, r_center_nocosmic['scores'], ax=ax1,
                                                title="center_Predictions")

                    fig.savefig("%i_%i_%i_%i.pdf"%(ev_pix.run(),ev_pix.subrun(),ev_pix.event(),ix), bbox_inches='tight')
                    '''                    

            tree.Fill()
            rd.reset_vertex()
        #entry += 1
    tfile.cd()
    tree.Write()
    tfile.Close()
    iom.finalize()

if __name__ == '__main__':
    
    if len(sys.argv) != 6:
        print
        print "\tIMAGE_FILE = str(sys.argv[1])"
        print "\tVTX_FILE   = str(sys.argv[2])"
        print "\tTRACK_FILE = str(sys.argv[3])"
        print "\tOUT_DIR    = str(sys.argv[4])"
        print "\tCFG        = str(sys.argv[5])"
        print 
        sys.exit(1)
    
    IMAGE_FILE = str(sys.argv[1]) 
    VTX_FILE   = str(sys.argv[2])
    TRACK_FILE = str(sys.argv[3])
    OUT_DIR    = str(sys.argv[4])
    CFG        = str(sys.argv[5])

    #CFG = os.path.join(BASE_PATH,"cfg","simple_config.cfg")

    with tf.device('/cpu:0'):
        main(IMAGE_FILE, VTX_FILE, TRACK_FILE, OUT_DIR, CFG)
    
    print "DONE!"
    sys.exit(0)
