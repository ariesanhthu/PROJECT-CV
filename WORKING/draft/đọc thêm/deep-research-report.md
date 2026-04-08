# Project Title

**CNN-Based Semantic Segmentation of Urban Scenes (Cityscapes Dataset)**

**Supervisor:** To be assigned (Dept. of Computer Science, University of Sydney)  
**Proposed Mode:** Data-driven deep learning research (supervised CNN model training)  

## Synopsis  
This project aims to develop and improve convolutional neural network (CNN) models for semantic segmentation of urban street scenes, using the Cityscapes dataset. Semantic segmentation (assigning a class label to every pixel) is critical for autonomous driving and city surveillance. State-of-the-art CNN architectures like PSPNet and DeepLabv3+ have achieved high accuracy on Cityscapes (80.2% and 82.1% mIoU respectively)【9†L69-L73】【23†L21-L24】. We will implement these baseline models and propose enhancements (e.g. advanced context modules, improved decoder structures) to further boost performance. The work will comprehensively evaluate CNN-only approaches (no Transformers) on Cityscapes, addressing challenges of high-resolution images and limited data. Preliminary analysis suggests CNN models can be competitive if carefully designed【9†L69-L73】【23†L21-L24】.

## Background  
Semantic segmentation in urban scenes has made rapid progress with deep learning.  Early work (FCN, 2015) showed that fully convolutional networks can be trained end-to-end for pixel-wise labeling, combining coarse semantic and fine appearance cues【47†L72-L77】.  Subsequent models introduced multi-scale context: Zhao et al. proposed PSPNet (Pyramid Scene Parsing Network) to aggregate regional context via pyramid pooling, achieving state-of-the-art results (80.2% mIoU on Cityscapes)【9†L63-L70】【9†L69-L73】.  Independently, Chen et al. developed DeepLabv3 which uses Atrous Spatial Pyramid Pooling (ASPP) and global image features to capture scale variance, significantly improving accuracy without post-processing【42†L69-L76】.  DeepLabv3+ extends this with an encoder-decoder architecture (adding a decoder to refine object boundaries)【23†L15-L18】, yielding 82.1% mIoU on Cityscapes【23†L21-L24】.  More recent improvements incorporate object-level context: for example, combining a high-resolution backbone (HRNet) with an Object-Contextual Representation (OCR) module achieved 85.4% mIoU on Cityscapes【32†L800-L806】.  

The **Cityscapes dataset** is a high-quality urban driving benchmark【17†L69-L73】【39†L494-L499】. It contains 5,000 finely annotated street-view images (and 20,000 coarse polygon annotations) from 50 European cities【17†L69-L73】. Each image is high-resolution (2048×1024) and labeled with 19 semantic classes (e.g. road, building, vehicle)【39†L494-L499】.  This dataset is well-suited for CNN-based segmentation since it represents real-world driving scenarios, but its moderate size (3k train images) means overfitting is a concern.  Recent work also highlights that CNN models can degrade under extreme conditions (weather, lighting)【38†L79-L82】, motivating robust training strategies. In summary, there is a strong literature on CNN segmentation (FCN, PSPNet, DeepLab, OCR, etc. above) and a clear benchmark (Cityscapes) which we will leverage and extend.

## Aims and Objectives  

- **Aim:** Achieve state-of-the-art CNN-based semantic segmentation on Cityscapes by implementing known architectures (PSPNet, DeepLabv3+) and developing novel enhancements tailored to urban scenes.

- **Objectives:**  
  1. **Literature Review:** Survey CNN segmentation methods (FCN, PSPNet, DeepLab, HRNet+OCR) and Cityscapes-related research【9†L63-L70】【23†L21-L24】.  
  2. **Data Preparation:** Set up the Cityscapes dataset (5,000 fine-labeled images, split into train/val/test【39†L494-L499】), and implement standard data augmentations (scaling, flipping, color jitter).  
  3. **Baseline Models:** Implement and train baseline CNN models: PSPNet and DeepLabv3+ with pretrained backbones (e.g. ResNet or Xception) as described in the literature【9†L69-L73】【23†L21-L24】. Use the standard evaluation split (2975 train, 500 val【39†L494-L499】) and measure segmentation accuracy with mean Intersection-over-Union (mIoU)【39†L505-L508】.  
  4. **Model Improvements:** Propose and integrate enhancements to the baselines. Examples include: adding an object-level context module (OCR) or attention mechanism (motivated by【32†L800-L806】), refining boundary detail with decoder networks【23†L15-L18】, using knowledge distillation to train lightweight CNNs, and leveraging coarse annotations for extra supervision.  
  5. **Experiments and Ablations:** Conduct controlled experiments to evaluate each enhancement. Compare models by mIoU and inference speed. Perform ablation studies (e.g. with/without context module) to isolate contributions.  
  6. **Analysis and Reporting:** Analyze which design choices most improve performance on urban scenes. Relate findings to the gap in existing work (e.g. robustness in adverse conditions【38†L79-L82】). Prepare comprehensive documentation (report and final proposal) of the methodology and results.

## Expected Contributions  

This research will provide a thorough comparison and enhancement of CNN-based segmentation methods for urban scenes, filling a gap where many recent works focus on vision transformers. Key expected contributions include:

- **Performance Benchmark:** A reproducible evaluation of PSPNet and DeepLabv3+ (and variations) on Cityscapes, showing current achievable accuracy (e.g. ~80–82% mIoU【9†L69-L73】【23†L21-L24】).  
- **Architectural Improvements:** A novel model variant incorporating advanced context aggregation (such as OCR【32†L800-L806】 or multi-scale attention) and refined decoding to boost segmentation quality, especially on object boundaries【23†L15-L18】.  
- **Robustness Insights:** Analysis of model robustness to dataset limitations and scene changes, addressing issues noted in recent studies (e.g. CNN fragility under extreme conditions【38†L79-L82】). This may include improved training strategies (augmentation, leveraging coarse labels).  
- **Contribution to Knowledge:** The findings will highlight how classic CNN segmentation models can be optimized for real-world urban environments (Cityscapes), offering guidance to the computer vision community on effective non-Transformer approaches.

## Proposed Methodology  

- **Dataset and Preprocessing:** Use the Cityscapes dataset【17†L69-L73】. Split the 5000 fine-labeled images into training (2975), validation (500) and test (1525) sets as per standard practice【39†L494-L499】. Preprocess images by resizing/cropping to manageable scales (e.g. 512×1024 inputs), and apply augmentations (random horizontal flips, scaling, color jitter) to combat overfitting. Optionally incorporate coarse-labeled images for extra weak supervision.  

- **Model Implementation:** Implement two baseline architectures: *PSPNet* (with a ResNet-50/101 backbone) and *DeepLabv3+* (with a ResNet or Xception backbone). Both models will be initialized with ImageNet-pretrained weights to leverage transfer learning. Use popular frameworks (PyTorch or TensorFlow). The DeepLabv3+ model includes Atrous Spatial Pyramid Pooling and a decoder to refine segmentation【23†L15-L18】.  

- **Training Procedure:** Train each model with supervised cross-entropy loss for 19 classes. Use stochastic gradient descent with momentum (or Adam) and a poly learning rate schedule. Evaluate on the validation set using mean IoU (average of per-class IoUs) and pixel accuracy【39†L505-L508】. Monitor overfitting given the dataset size.  

- **Model Enhancements:** Explore enhancements such as:  
  - **Context Modules:** Integrate an Object-Contextual Representation (OCR) module or self-attention block after the backbone, to better capture class relationships【32†L800-L806】.  
  - **Decoder Refinement:** Extend the decoder (as in DeepLabv3+) by adding skip connections or boundary refinement blocks to sharpen edges【23†L15-L18】.  
  - **Lightweight Backbones:** Experiment with more efficient backbones (e.g. MobileNet or ResNet-18) and apply knowledge distillation from larger teacher networks (e.g. PSPNet-R101) to improve smaller models.  
  - **Data Augmentation:** Use advanced augmentations (style transfer, random weather effects) to simulate variations and improve robustness. This is motivated by findings that model performance drops in extreme scenes【38†L79-L82】.  

- **Evaluation:** After training, evaluate each variant on the Cityscapes validation and test sets. Report metrics including mean IoU and class-wise IoU. Compare against baseline performance and published state-of-art. Use visualization (segmentation maps) to qualitatively assess improvements.  

## Work Plan  

A tentative timeline (assuming a 12-month project) is:  
- **Months 1–2:** Literature review; dataset setup; baseline code preparation.  
- **Months 3–4:** Implement PSPNet and DeepLabv3+; run initial training; debug pipeline.  
- **Month 5:** Evaluate baselines on validation set; identify weaknesses (e.g., errors on small classes or boundaries).  
- **Months 6–7:** Develop and integrate improvements (OCR module, decoder tweaks, distillation, etc.); implement additional augmentations.  
- **Months 8–9:** Conduct experiments on improved models; perform hyperparameter tuning and ablation studies.  
- **Month 10:** Final evaluation on test set; compare all models; analyze results.  
- **Months 11–12:** Compile results; write the research report and proposal sections; prepare final documentation.  

Key milestones include completing baseline training, implementing major enhancements, and achieving a performance gain over baselines.

## Resources  

- **Data:** Cityscapes dataset (publicly available, requires ~100GB storage for high-resolution images and annotations)【17†L69-L73】.  
- **Hardware:** Access to a GPU-equipped workstation or cloud GPU instances (e.g. NVIDIA RTX 3080 or equivalent) for training deep CNNs. Training may require dozens of GPU-hours depending on model size.  
- **Software:** Deep learning framework (PyTorch or TensorFlow), with libraries for image processing and evaluation. Cityscapes utility scripts for data loading.  
- **Miscellaneous:** Standard workstation, sufficient RAM (32GB+), and a stable development environment. No specialized equipment beyond a GPU is needed.  

Budget considerations are minimal, as the project primarily uses open datasets and available computing resources. 

## References

- Cordts *et al.*, “The Cityscapes Dataset for Semantic Urban Scene Understanding”【17†L69-L73】. *arXiv:1604.01685* (2016).  
- Zhao *et al.*, “Pyramid Scene Parsing Network” (PSPNet)【9†L69-L73】. *CVPR 2017*.  
- Chen *et al.*, “Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation” (DeepLabv3+)【23†L21-L24】. *ECCV 2018*.  
- Chen *et al.*, “Rethinking Atrous Convolution for Semantic Image Segmentation” (DeepLabv3)【42†L69-L76】. *arXiv:1706.05587* (2017).  
- Long *et al.*, “Fully Convolutional Networks for Semantic Segmentation”【47†L72-L77】. *CVPR 2015*.  
- Yuan *et al.*, “Object-Contextual Representations for Semantic Segmentation” (OCR+HRNet)【32†L800-L806】. *ECCV 2020*.  
- Suryanto *et al.*, “Cityscape-Adverse: Benchmarking Robustness of Semantic Segmentation…”【38†L79-L82】. *arXiv:2411.00425* (2024).