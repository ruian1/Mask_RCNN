import ROOT
import numpy as np
import bbox_helper as bh
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
def Mask_based_analyze ( rd, name, mrcnn_result, projected_tracks_contours):
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
                            
def Vertex_based_analyze (rd, name, mrcnn_result, x_2d, y_2d):

    for each in mrcnn_result['scores']:
        rd_scores_plane2 = getattr(rd, '%s_scores_plane2'%name)
        rd_scores_plane2.push_back(each)
        
    for each in mrcnn_result['class_ids']:
        rd_class_ids_plane2 = getattr(rd, '%s_class_ids_plane2'%name)
        rd_class_ids_plane2.push_back(class_names[each])
        #rd.center_class_ids_plane2.push_back(class_names[each])
    
    for x in xrange(mrcnn_result['rois'].shape[0]):
        roi_int=ROOT.std.vector("int")(4,0)
        roi_int.clear()
        for roi_int32 in mrcnn_result['rois'][x]:
            roi_int.push_back(int(roi_int32))

        rd_rois_plane2 = getattr(rd, '%s_rois_plane2'%name)
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
    for each_class in classes_np :
        pdg=class_names[each_class]
        if pdg==11:
            this_sum=np.sum(masks_np[idx])
            rd_electron_mask_sum = getattr(rd, '%s_electron_mask_sum'%name)
            rd_electron_mask_sum.push_back(np.int(this_sum))
            #rd.center_electron_mask_sum.push_back(np.int(this_sum))

            this_dist=bh.Point_points_distances([x_2d, y_2d], np.nonzero(masks_np[idx].reshape(512,512)))

            rd_electron_mask_dist = getattr(rd, '%s_electron_mask_dist'%name)
            rd_electron_mask_dist.push_back(np.int(this_dist))
            #rd.center_electron_mask_dist.push_back(this_dist)

        elif pdg==13:
            this_sum=np.sum(masks_np[idx])
            
            rd_muon_mask_sum = getattr(rd, '%s_muon_mask_sum'%name)
            rd_muon_mask_sum.push_back(np.int(this_sum))

            #rd.center_muon_mask_sum.push_back(np.int(this_sum))
            this_dist=bh.Point_points_distances([x_2d, y_2d], np.nonzero(masks_np[idx].reshape(512,512)))
            rd_muon_mask_dist = getattr(rd, '%s_muon_mask_dist'%name)
            rd_muon_mask_dist.push_back(np.int(this_dist))
            #rd.center_muon_mask_dist.push_back(this_dist)

        elif pdg==2212:
            this_sum=np.sum(masks_np[idx])

            rd_proton_mask_sum = getattr(rd, '%s_proton_mask_sum'%name)
            rd_proton_mask_sum.push_back(np.int(this_sum))
            #rd.center_proton_mask_sum.push_back(np.int(this_sum))
            this_dist=bh.Point_points_distances([x_2d, y_2d], np.nonzero(masks_np[idx].reshape(512,512)))
            rd_proton_mask_dist = getattr(rd, '%s_proton_mask_dist'%name)
            rd_proton_mask_dist.push_back(np.int(this_dist))
            #rd.center_proton_mask_dist.push_back(this_dist)

        idx+=1
            

                    

