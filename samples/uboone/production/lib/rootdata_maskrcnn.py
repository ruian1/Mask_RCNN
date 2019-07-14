import ROOT
from array import array

kINVALID_INT    = ROOT.std.numeric_limits("int")().lowest()
kINVALID_FLOAT  = ROOT.std.numeric_limits("float")().lowest()
kINVALID_DOUBLE = ROOT.std.numeric_limits("double")().lowest()
kINVALID_BOOL   = ROOT.std.numeric_limits("bool")().lowest()


class ROOTData(object):

    def __init__(self):
        self.run    = array( 'i', [ kINVALID_INT ] )
        self.subrun = array( 'i', [ kINVALID_INT ] )
        self.event  = array( 'i', [ kINVALID_INT ] )
        self.entry  = array( 'i', [ kINVALID_INT ] )
        self.vtxid  = array( 'i', [ kINVALID_INT ] )
        self.num_vertex = array( 'i', [ kINVALID_INT ] )
        
        self.inferred = array( 'i', [ kINVALID_INT ] )


        # To-do, write sparse matrix, also needs changes in larcv
        #self.masks_plane2_1d  = ROOT.std.vector(ROOT.std.vector("bool"))(3, ROOT.std.vector("bool")(262144, kINVALID_BOOL))

        self.center_scores_plane2      = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.center_class_ids_plane2   = ROOT.std.vector("int")(3,kINVALID_INT)
        self.center_rois_plane2        = ROOT.std.vector(ROOT.std.vector("int"))(3, ROOT.std.vector("int")(4, kINVALID_INT))
        self.center_electron_mask_sum  = ROOT.std.vector("int")(3, kINVALID_INT)
        self.center_muon_mask_sum      = ROOT.std.vector("int")(3, kINVALID_INT)
        self.center_proton_mask_sum    = ROOT.std.vector("int")(3, kINVALID_INT)
        self.center_electron_mask_dist = ROOT.std.vector("float")(3, kINVALID_FLOAT)
        self.center_muon_mask_dist     = ROOT.std.vector("float")(3, kINVALID_FLOAT)
        self.center_proton_mask_dist   = ROOT.std.vector("float")(3, kINVALID_FLOAT)

        self.pix_scores_plane2      = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.pix_class_ids_plane2   = ROOT.std.vector("int")(3,kINVALID_INT)
        self.pix_rois_plane2        = ROOT.std.vector(ROOT.std.vector("int"))(3, ROOT.std.vector("int")(4, kINVALID_INT))
        self.pix_electron_mask_sum  = ROOT.std.vector("int")(3, kINVALID_INT)
        self.pix_muon_mask_sum      = ROOT.std.vector("int")(3, kINVALID_INT)
        self.pix_proton_mask_sum    = ROOT.std.vector("int")(3, kINVALID_INT)
        self.pix_electron_mask_dist = ROOT.std.vector("float")(3, kINVALID_FLOAT)
        self.pix_muon_mask_dist     = ROOT.std.vector("float")(3, kINVALID_FLOAT)
        self.pix_proton_mask_dist   = ROOT.std.vector("float")(3, kINVALID_FLOAT)

        self.int_scores_plane2      = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.int_class_ids_plane2   = ROOT.std.vector("int")(3,kINVALID_INT)
        self.int_rois_plane2        = ROOT.std.vector(ROOT.std.vector("int"))(3, ROOT.std.vector("int")(4, kINVALID_INT))
        self.int_electron_mask_sum  = ROOT.std.vector("int")(3, kINVALID_INT)
        self.int_muon_mask_sum      = ROOT.std.vector("int")(3, kINVALID_INT)
        self.int_proton_mask_sum    = ROOT.std.vector("int")(3, kINVALID_INT)
        self.int_electron_mask_dist = ROOT.std.vector("float")(3, kINVALID_FLOAT)
        self.int_muon_mask_dist     = ROOT.std.vector("float")(3, kINVALID_FLOAT)
        self.int_proton_mask_dist   = ROOT.std.vector("float")(3, kINVALID_FLOAT)

        # Store trackid->pids for each track
        self.mask_pids_array = ROOT.std.vector(ROOT.std.vector("int"))(3, ROOT.std.vector("int")(4, kINVALID_INT))
        # Stote best pid for each track
        self.mask_pids = ROOT.std.vector("float")(3, kINVALID_FLOAT)
        
        
    def reset_event(self):
        self.run[0]     = kINVALID_INT
        self.subrun[0]  = kINVALID_INT
        self.event[0]   = kINVALID_INT
        self.entry[0]   = kINVALID_INT

        self.num_vertex[0] = kINVALID_INT
        
    def reset_vertex(self):
        self.vtxid[0]   = kINVALID_INT
        self.inferred[0] = kINVALID_INT

        # Vertex based analyze
        self.center_scores_plane2.clear()
        self.center_class_ids_plane2.clear()
        self.center_rois_plane2.clear()
        self.center_electron_mask_sum.clear()
        self.center_muon_mask_sum.clear()
        self.center_proton_mask_sum.clear()
        self.center_electron_mask_dist.clear()
        self.center_muon_mask_dist.clear()
        self.center_proton_mask_dist.clear()

        self.pix_scores_plane2.clear()
        self.pix_class_ids_plane2.clear()
        self.pix_rois_plane2.clear()        
        self.pix_electron_mask_sum.clear()
        self.pix_muon_mask_sum.clear()
        self.pix_proton_mask_sum.clear()
        self.pix_electron_mask_dist.clear()
        self.pix_muon_mask_dist.clear()
        self.pix_proton_mask_dist.clear()

        self.int_scores_plane2.clear()
        self.int_class_ids_plane2.clear()
        self.int_rois_plane2.clear()        
        self.int_electron_mask_sum.clear()
        self.int_muon_mask_sum.clear()
        self.int_proton_mask_sum.clear()
        self.int_electron_mask_dist.clear()
        self.int_muon_mask_dist.clear()
        self.int_proton_mask_dist.clear()

        # Mask-based analyze
        self.mask_pids_array.clear()
        self.mask_pids.clear()
        
    def reset(self):
        self.reset_event()
        self.reset_vertex()

    def init_tree(self,tree):
        
        tree.Branch("run"   , self.run   , "run/I")
        tree.Branch("subrun", self.subrun, "subrun/I")
        tree.Branch("event" , self.event , "event/I")
        tree.Branch("entry" , self.entry , "entry/I")

        tree.Branch("vtxid" , self.vtxid, "vtxid/I")

        tree.Branch("num_vertex" , self.num_vertex, "num_vertex/I")

        tree.Branch("inferred"   , self.inferred  , "inferred/I")

        tree.Branch("center_scores_plane2", self.center_scores_plane2)
        tree.Branch("center_class_ids_plane2", self.center_class_ids_plane2)
        tree.Branch("center_rois_plane2", self.center_rois_plane2)
        tree.Branch("center_electron_mask_sum", self.center_electron_mask_sum)
        tree.Branch("center_muon_mask_sum", self.center_muon_mask_sum)
        tree.Branch("center_proton_mask_sum", self.center_proton_mask_sum)
        tree.Branch("center_electron_mask_dist", self.center_electron_mask_dist)
        tree.Branch("center_muon_mask_dist", self.center_muon_mask_dist)
        tree.Branch("center_proton_mask_dist", self.center_proton_mask_dist)

        tree.Branch("pix_scores_plane2", self.pix_scores_plane2)
        tree.Branch("pix_class_ids_plane2", self.pix_class_ids_plane2)
        tree.Branch("pix_rois_plane2", self.pix_rois_plane2)
        tree.Branch("pix_electron_mask_sum", self.pix_electron_mask_sum)
        tree.Branch("pix_muon_mask_sum", self.pix_muon_mask_sum)
        tree.Branch("pix_proton_mask_sum", self.pix_proton_mask_sum)
        tree.Branch("pix_electron_mask_dist", self.pix_electron_mask_dist)
        tree.Branch("pix_muon_mask_dist", self.pix_muon_mask_dist)
        tree.Branch("pix_proton_mask_dist", self.pix_proton_mask_dist)

        tree.Branch("int_scores_plane2", self.int_scores_plane2)
        tree.Branch("int_class_ids_plane2", self.int_class_ids_plane2)
        tree.Branch("int_rois_plane2", self.int_rois_plane2)        
        tree.Branch("int_electron_mask_sum", self.int_electron_mask_sum)
        tree.Branch("int_muon_mask_sum", self.int_muon_mask_sum)
        tree.Branch("int_proton_mask_sum", self.int_proton_mask_sum)
        tree.Branch("int_electron_mask_dist", self.int_electron_mask_dist)
        tree.Branch("int_muon_mask_dist", self.int_muon_mask_dist)
        tree.Branch("int_proton_mask_dist", self.int_proton_mask_dist)

        # Mask based analyze
        tree.Branch("mask_pids_array", self.mask_pids_array)
        tree.Branch("mask_pids", self.mask_pids)
        
