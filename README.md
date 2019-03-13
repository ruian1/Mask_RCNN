# Mask R-CNN for LArTPC 

Applying [Mask R-CNN by matterport](https://github.com/matterport/Mask_RCNN) on Liquid Argon Time Projection Chamber (LArTPC) data to help charge clustering and particle identification. LArTPC is the next generation dectector used in a series of neutrino experiments, [Short Baseline Neutrino program](https://sbn.fnal.gov/) (including [MicroBooNE](https://microboone.fnal.gov/), [SBND](http://sbn-nd.fnal.gov/), [ICARUS](https://icarus.fnal.gov/)), [ProtoDune](https://www.symmetrymagazine.org/article/protodune-in-pictures) and [Dune](https://lbnf.fnal.gov/) (long baseline neutrino experiment with 2,000 ton LArTPC). 

Data of particle interactiond in LArTPCs is recored as deposited charge over the time tick and readout wires, forming a clean representation as 2D image, perfect for deep learning studies. 

![Instance Segmentation Sample](assets/profile_mrcnn_ex.png)

