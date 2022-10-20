# Discriminative, Restorative, and Adversarial Learning: Stepwise Incremental Pretraining.
This repository provides a Keras implementation of the Discriminative, Restorative, and Adversarial Learning: Stepwise Incremental Pretraining

We extend and reimplement five prominent self-supervised learning methods and integrate them into a united framework that incorporates three learning components: Discriminative, Restorative, and Adversarial Learning. We then explore the stepwise training strategies that stabilize the united framework's training process and improve the overall performance for the target tasks.


<div>
<img align=center width="20%" src="IMAGE/logo_gif.gif" /><img align=center width="80%" src="IMAGE/United.png" />
</div>

The five self-supervised learning methods are Jigsaw, Rubik's Cube & Rubik's Cube++，Rotation, Deep CLustering, TransVW.

<p>
<div>
<img align=center width="50%" src="IMAGE/Jigsaw.png" /><img align=center width="50%" src="IMAGE/Rubik'sCube.png" />
  <img align=center width="50%" src="IMAGE/Rotation.png" /><img align=center width="50%" src="IMAGE/DeepCluster.png" />
  <img align=left width="100%" src="IMAGE/TransVW.png" />
</div>
</p>



## Publication
<b>Discriminative, Restorative, and Adversarial Learning: Stepwise Incremental Pretraining </b> <br/>
[Zuwei Guo](https://github.com/AbhorsenKnight)<sup>1</sup>, [Nahid Ul Islam](https://github.com/Nahid1992)<sup>1</sup>, [Michael B. Gotway](https://www.mayoclinic.org/biographies/gotway-michael-b-m-d/bio-20055566)<sup>2</sup>, [Jianming Liang](https://chs.asu.edu/jianming-liang)<sup>1</sup><br/>
<sup>1 </sup>Arizona State University, <sup>2 </sup>Mayo Clinic <br/>
Published in: **Domain Adaptation and Representation Transfer (DART), 2022.**

[Paper](#) | [Code](https://github.com/jlianglab/StepwisePretraining) | [Poster] | [Slides] | Presentation ([YouTube])



## Major results from our work
1. We found the optimum pretraining strategy for the United framework
Stepwise pretraining is always better than training everything together
<div>
  <img align=center width="100%" src="IMAGE/Optimum_pretrain.png" />
</div>

2. We found the effective utilization for pretrained components for target tasks.
a) For classification tasks, pretrained encoders perform much better than the randomly initialized encoders
<div>
  <img align=center width="50%" src="IMAGE/NCC_D.png" /><img align=center width="50%" src="IMAGE/ECC_D.png" />
</div>

b) For segmentation tasks, the trained encoder also improves performance for most methods. But we also observed some negative transfer due to task mismatches.
<div>
  <img align=center width="33%" src="IMAGE/NCS_D.png" /><img align=center width="33%" src="IMAGE/LCS_D.png" /><img align=center width="33%" src="IMAGE/BMS_D.png" />
</div>

c) For segmentation tasks, we should always transfer pretrained decoders.
<div>
  <img align=center width="33%" src="IMAGE/Jigsaw_DR.png" /><img align=center width="33%" src="IMAGE/Rubik_DR.png" /><img align=center width="33%" src="IMAGE/DeepCluster_DR.png" />
  <img align=center width="33%" src="IMAGE/TransVW_DR.png" /><img align=center width="33%" src="IMAGE/Rotation_DR.png" />
</div>

4) Adversarial training ((D)+R)+A strengthens learned representation
<div>
  <img align=center width="100%" src="IMAGE/ALL_DRA.png" />
</div>

5) Adversarial training ((D)+R)+A reduces annotation costs
<div>
  <img align=center width="100%" src="IMAGE/Fewer_label.png" />
</div>



## Acknowledgement
This research has been supported in part by ASU and Mayo Clinic through a Seed Grant and an Innovation Grant, and in part by the NIH under Award Number R01HL128785. The content is solely the responsi- bility of the authors and does not necessarily represent the official views of the NIH. This work has utilized the GPUs provided in part by the ASU Research Computing and in part by the Extreme Science and Engineering Discovery Environment (XSEDE) funded by the National Science Foundation (NSF) under grant numbers: ACI-1548562, ACI-1928147, and ACI-2005632. The content of this paper is covered by patents pending.



## License

Released under the [ASU GitHub Project License](./LICENSE).
