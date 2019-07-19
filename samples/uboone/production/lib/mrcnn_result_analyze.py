import ROOT
import numpy as np
import bbox_helper as bh
import point_helper as ph
import cv2

class_names=[0, 11, 22, 13, 211, 2212, 2213]

def Mrcnn_clasid_2_output_classid(mrcnn_classid):

    # 0, pdg 11
    if mrcnn_classid == 1:
        return 0

    # 1, pdg 13
    if mrcnn_classid == 3:
        return 1

    # 2, pdg 211
    if mrcnn_classid == 4:
        return 2

    # 3, pdg 2212
    if mrcnn_classid == 5:
        return 3

#def Mask_based_analyze (name, mrcnn_result, projected_tracks_contours):
def Mask_based_analyze ( rd, branch_name, mrcnn_result, projected_tracks_contours):
    if not len(projected_tracks_contours):
        return True

    if not (len(mrcnn_result['class_ids'])):
        return True
    
    # Output array track_id -> track pid
    # Each track has a 4 length score which is sum of masks
    # of one PDG from [11,13,211,2212] corresponding to
    # [1,3,4,5] in class_ids
    #track_pid_arrays = np.zeros((len(projected_tracks_contours), 4))

    track_pid_arrays = {}
    
    r_masks = mrcnn_result['masks']
    r_class_ids = mrcnn_result['class_ids']
    
    for track_id, tracks_contours in projected_tracks_contours.iteritems():
        
        pid_array = np.zeros(4)
        #print "track_id, tracks_contours", track_id, tracks_contours
        
        # Loop over contours found on a reco track
        for track_contour in tracks_contours:
            for r_idx in xrange(r_class_ids.shape[0]):
                class_idx_0123 = Mrcnn_clasid_2_output_classid(r_class_ids[r_idx])
                mask_pt = np.argwhere(r_masks[:, :, r_idx]==1)
                for each in mask_pt:
                    #pt = (np.float64(each[1]), np.float64(each[0]))
                    pt = (ROOT.Double(each[1]), ROOT.Double(each[0]))
                                        
                    #print pt, cv2.pointPolygonTest(track_contour,pt,False)
                    
                    # If a mask pt on/inside a contour, sum it
                    if (cv2.pointPolygonTest(track_contour,pt,False) >= 0 ):
                        pid_array[class_idx_0123] += 1

        track_pid_arrays[track_id] = pid_array

        pid_vec=ROOT.std.vector("int")(4,0)
        pid_vec.clear()

        for pid in pid_array:
            pid_vec.push_back(np.int(pid))
        rd.mask_pids_array.push_back(pid_vec)

        if (np.sum(pid_array)):
            rd.mask_pids.push_back(np.argmax(pid_vec))
        else:
            rd.mask_pids.push_back(-1)
    #return track_pid_arrays
                            
def Vertex_based_analyze (rd, branch_name, mrcnn_result, x_2d, y_2d, verbose = 0):

    for each in mrcnn_result['scores']:
        rd_scores_plane2 = getattr(rd, '%s_scores_plane2'%branch_name)
        rd_scores_plane2.push_back(each)
        
    for each in mrcnn_result['class_ids']:
        rd_class_ids_plane2 = getattr(rd, '%s_class_ids_plane2'%branch_name)
        rd_class_ids_plane2.push_back(class_names[each])
        #rd.center_class_ids_plane2.push_back(class_names[each])
    
    for x in xrange(mrcnn_result['rois'].shape[0]):
        roi_int=ROOT.std.vector("int")(4,0)
        roi_int.clear()
        for roi_int32 in mrcnn_result['rois'][x]:
            roi_int.push_back(int(roi_int32))

        rd_rois_plane2 = getattr(rd, '%s_rois_plane2'%branch_name)
        rd_rois_plane2.push_back(roi_int)
        #rd.center_rois_plane2.push_back(roi_int)

    classes_np=mrcnn_result['class_ids']
    # masks are too large, now only store needed values
    masks_np=np.zeros([mrcnn_result['masks'].shape[-1], 512 * 512])
    
    for x in xrange(mrcnn_result['masks'].shape[-1]):
        this_mask=mrcnn_result['masks'][:,:,x]
        this_mask=this_mask.flatten()
        masks_np[x] = this_mask
        
    idx=0
    particle_names = {11:"electron", 13:"muon", 2212:"proton", 211:"pion"}
    for each_class in classes_np :
        pdg=class_names[each_class]
        if(verbose): print "================"
        if(verbose): print "pdg, ", pdg
        Calculate_mask_and_dist (masks_np, idx, rd, branch_name, particle_names[pdg], x_2d, y_2d)

        idx+=1

def Calculate_mask_and_dist (masks_np, idx, rd, branch_name, particle_name, x_2d, y_2d, verbose = 0):
    this_sum=np.sum(masks_np[idx])

    rd_particle_mask_sum = getattr(rd, '%s_%s_mask_sum'%(branch_name, particle_name))
    rd_particle_mask_sum.push_back(np.int(this_sum))
    
    rd_particle_mask_dist = getattr(rd, '%s_%s_mask_dist'%(branch_name, particle_name))
    rd_particle_mask_dist_simple = getattr(rd, '%s_%s_mask_dist_simple'%(branch_name, particle_name))

    
    this_dist=bh.Point_points_distances([x_2d, y_2d], np.nonzero(masks_np[idx].reshape(512,512)), xyswap = 1)    
    rd_particle_mask_dist_simple.push_back(this_dist)
    if(verbose): print "vtx 2d is ", x_2d, y_2d
    if(verbose): print "min pt to pts dist is ", this_dist
    
    if(verbose): print "the pt is ", bh.Point_points_distances([x_2d, y_2d], np.nonzero(masks_np[idx].reshape(512,512)), mode=1, xyswap = 1)
    # farthest pt to vertex
    
    farthest_pt = bh.Point_points_distances([x_2d, y_2d], np.nonzero(masks_np[idx].reshape(512,512)), mode=1, xyswap = 1)
    if farthest_pt == -999:
        rd_particle_mask_dist.push_back(np.int(farthest_pt))
        return
        
    # the farthest pt to farthest pt to vertex
    farthest_pt_other = bh.Point_points_distances(farthest_pt, np.nonzero(masks_np[idx].reshape(512,512)), mode=1)
    # return the smaller one between two farthest points
    min_dist = min(ph.Dist_pt_to_pt(np.array([y_2d, x_2d]), farthest_pt),
                   ph.Dist_pt_to_pt(np.array([y_2d, x_2d]), farthest_pt_other))
    if(verbose): print "    farthest pt is,", farthest_pt
    if(verbose): print "2dn farthest pt is,", farthest_pt_other
    
    if(verbose): print "new closest dist is ", min_dist
    
    rd_particle_mask_dist.push_back(np.int(min_dist))


    
    
