# FIESTA

This repository provides a PyTorch implementation of the following paper:
> **FIESTA: Fourier-Based Semantic Augmentation with Uncertainty Guidance for Enhanced Domain Generalizability in Medical Image Segmentation**<br>
> [Kwanseok Oh](https://scholar.google.co.kr/citations?user=EMYHaHUAAAAJ&hl=ko)<sup>1</sup>, [Eunjin Jeon](https://scholar.google.co.kr/citations?user=U_hg5B0AAAAJ&hl=ko)<sup>2</sup>, [Da-Woon Heo](https://scholar.google.co.kr/citations?user=WapMdZ8AAAAJ&hl=ko&oi=ao)<sup>1</sup>, [Yooseung Shin](https://scholar.google.co.kr/citations?user=yCvN9Z8AAAAJ&hl=ko)<sup>1</sup>, and [Heung-Il Suk](https://scholar.google.co.kr/citations?user=dl_oZLwAAAAJ&hl=ko)<sup>1, 2</sup><br/>
> (<sup>1</sup>Department of Artificial Intelligence, Korea University) <br/>
> (<sup>2</sup>Department of Brain and Cognitive Engineering, Korea University) <br/>
> Official Version: https://arxiv.org/abs/2406.14308 <br/>
> 
> **Abstract:** *Single-source domain generalization (SDG) in medical image segmentation (MIS) aims to generalize a model using data from only one source domain to segment data from an unseen target domain. Despite substantial advances in SDG with data augmentation, existing methods often fail to fully consider the details and uncertain areas prevalent in MIS, leading to mis-segmentation. This paper proposes a Fourier-based semantic augmentation method called FIESTA using uncertainty guidance to enhance the fundamental goals of MIS in an SDG context by manipulating the amplitude and phase components in the frequency domain. The proposed Fourier augmentative transformer addresses semantic amplitude modulation based on meaningful angular points to induce pertinent variations and harnesses the phase spectrum to ensure structural coherence. Moreover, FIESTA employs epistemic uncertainty to fine-tune the augmentation process, improving the ability of the model to adapt to diverse augmented data and concentrate on areas with higher ambiguity. Extensive experiments across three cross-domain scenarios demonstrate that FIESTA surpasses recent state-of-the-art SDG approaches in segmentation performance and significantly contributes to boosting the applicability of the model in medical imaging modalities.*

## Overall Framework
- To the best of our knowledge, this work is the first Fourier-based augmentation method that simultaneously manipulates amplitude and phase components using meaningful factors tailored to SDG for cross-domain MIS.
- We propose the FAT, providing an advanced augmentation strategy that combines masking and modulation techniques to transform the amplitude spectrum and applies filtering to refine the phase information to impose structural integrity.
- The FIESTA framework embraces an uncertainty-guided mutual augmentation strategy by applying UG to focus learning in the segmentation model on certain areas of high ambiguity or mis-segmented locations.
- Based on the quantitative and qualitative experimental results on various cross-domain scenarios (including cross-modality, cross-sequence, and cross-sites), we demonstrate the significant robustness and generalizability of FIESTA, which surpasses state-of-the-art SDG methods.

<p align="center"><img width="90%" src="https://github.com/ku-milab/LiCoL/assets/57162425/17ef8f9e-d315-4b13-b3b8-eed65f1f2ecd" /></p>

## Qualitative Analyses
### Illustration of inferred AD-effect and statistical maps in binary and multi-class scenarios
<p align="center"><img width="100%" src="https://github.com/ku-milab/LiCoL/assets/57162425/d9fae7e4-a506-4fdd-b36f-19f7bb29b54f" /></p>
<p align="center"><img width="65%" src="https://github.com/ku-milab/LiCoL/assets/57162425/532c56a6-d412-4822-b89b-1e3d8b6b3c0a" /></p>

### Visualization of a normalized AD-relatedness index over the group-wise (first column) and individuals (second and third columns) 
<p align="center"><img width="100%" src="https://github.com/ku-milab/LiCoL/assets/57162425/2168d64a-6c21-46bc-a6c6-2deabd9ae2be" /></p>

## Quantitative Evaluations
<p align="center"><img width="100%" src="https://github.com/ku-milab/LiCoL/assets/57162425/7438832c-9fff-48ea-a46c-ba6306e8c7e4" /></p>
<p align="center"><img width="50%" src="https://github.com/ku-milab/LiCoL/assets/57162425/44d146ef-5884-4f98-9f8a-6f50ee7ef060" /></p>

## Requirements
* [TensorFlow 2.2.0+](https://www.tensorflow.org/)
* [Python 3.6+](https://www.continuum.io/downloads)
* [Scikit-learn 0.23.2+](https://scikit-learn.org/stable/)
* [Nibabel 3.0.1+](https://nipy.org/nibabel/)

## Downloading datasets
To download Alzheimer's disease neuroimaging initiative dataset
* https://adni.loni.usc.edu/

## How to Run
### Counterfactual Map Generation
Mode: #0 Learn, #1 Explain

1. Learn: pre-training the predictive model
>- `CMG_config.py --mode="Learn"`
>- Set the mode as a "Learn" to train the predictive model

2. Explain: Counterfactual map generation using a pre-trained diagnostic model
>- `CMG_config.py --mode="Explain" --dataset=None --scenario=None`
>- Change the mode from "Learn" to "Explain" on Config.py
>- Set the classifier and encoder weight for training (freeze)
>- Set the variables of dataset and scenario for training

### AD-Effect Map and LiCoL
1. AD-effect map acquisition based on manipulated real-/counterfactual-labeled gray matter density maps
>- `AD-effect Map Acquisition.ipynb`
>- This step for the AD-effect map acquisition was implemented by using the Jupyter notebook
>- Execute markdown cells written in jupyter notebook in order

2. LiCoL
>- `LiCoL_ALL.py --datatset=None --scenario=None --data_path==None`
>- Set the variables of dataset and scenario for training
>- For example, dataset="ADNI" and scenario="CN_AD"
>- Modify the data path for uploading the dataset (=line 234)

## Citation
If you find this work useful for your research, please cite the following paper:

```
@article{oh2023quantitatively,
  title={A Quantitatively Interpretable Model for Alzheimer's Disease Prediction Using Deep Counterfactuals},
  author={Oh, Kwanseok and Heo, Da-Woon and Mulyadi, Ahmad Wisnu and Jung, Wonsik and Kang, Eunsong and Lee, Kun Ho and Suk, Heung-Il},
  journal={arXiv preprint arXiv:2310.03457},
  year={2023}
}
```

## Acknowledgement
This work was supported by the Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) No. 20220-00959 ((Part 2) Few-Shot Learning of Causal Inference in Vision and Language for Decision Making) and No. 20190-00079 (Department of Artificial Intelligence (Korea University)). This study was further supported by KBRI basic research program through Korea Brain Research Institute funded by the Ministry of Science and ICT (22-BR-03-05).
