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

        self.scores_plane2    = ROOT.std.vector("float")(3,kINVALID_FLOAT)
        self.class_ids_plane2 = ROOT.std.vector("int")(3,kINVALID_INT)
        self.rois_plane2      = ROOT.std.vector(ROOT.std.vector("int"))(3, ROOT.std.vector("int")(4, kINVALID_INT))

        # To-do, write sparse matrix, also needs changes in larcv
        #self.masks_plane2_1d  = ROOT.std.vector(ROOT.std.vector("bool"))(3, ROOT.std.vector("bool")(262144, kINVALID_BOOL))

        self.electron_mask_sum = ROOT.std.vector("int")(3, kINVALID_INT)
        self.muon_mask_sum = ROOT.std.vector("int")(3, kINVALID_INT)
        self.proton_mask_sum = ROOT.std.vector("int")(3, kINVALID_INT)
        
        self.electron_mask_dist = ROOT.std.vector("float")(3, kINVALID_FLOAT)
        self.muon_mask_dist = ROOT.std.vector("float")(3, kINVALID_FLOAT)
        self.proton_mask_dist = ROOT.std.vector("float")(3, kINVALID_FLOAT)

    def reset_event(self):
        self.run[0]     = kINVALID_INT
        self.subrun[0]  = kINVALID_INT
        self.event[0]   = kINVALID_INT
        self.entry[0]   = kINVALID_INT

        self.num_vertex[0] = kINVALID_INT
        
    def reset_vertex(self):
        self.vtxid[0]   = kINVALID_INT
        self.inferred[0] = kINVALID_INT

        self.scores_plane2.clear()
        self.class_ids_plane2.clear()
        self.rois_plane2.clear()
        #self.masks_plane2_1d.clear()
        #self.masks_plane2_2d.clear()

        self.electron_mask_sum.clear()
        self.muon_mask_sum.clear()
        self.proton_mask_sum.clear()
        
        self.electron_mask_dist.clear()
        self.muon_mask_dist.clear()
        self.proton_mask_dist.clear()
        
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

        tree.Branch("scores_plane2", self.scores_plane2)
        tree.Branch("class_ids_plane2", self.class_ids_plane2)
        tree.Branch("rois_plane2", self.rois_plane2)
        #tree.Branch("masks_plane2_2d", self.masks_plane2_2d)
        #tree.Branch("masks_plane2_1d", self.masks_plane2_1d)

        tree.Branch("electron_mask_sum", self.electron_mask_sum)
        tree.Branch("muon_mask_sum", self.muon_mask_sum)
        tree.Branch("proton_mask_sum", self.proton_mask_sum)

        tree.Branch("electron_mask_dist", self.electron_mask_dist)
        tree.Branch("muon_mask_dist", self.muon_mask_dist)
        tree.Branch("proton_mask_dist", self.proton_mask_dist)

        
