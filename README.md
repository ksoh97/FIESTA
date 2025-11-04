# FIESTA

This repository provides a PyTorch implementation of the following paper:
> **FIESTA: Fourier-Based Semantic Augmentation with Uncertainty Guidance for Enhanced Domain Generalizability in Medical Image Segmentation**<br>
> [Kwanseok Oh](https://scholar.google.co.kr/citations?user=EMYHaHUAAAAJ&hl=ko)<sup>1</sup>, [Eunjin Jeon](https://scholar.google.co.kr/citations?user=U_hg5B0AAAAJ&hl=ko)<sup>2, 3</sup>, [Da-Woon Heo](https://scholar.google.co.kr/citations?user=WapMdZ8AAAAJ&hl=ko&oi=ao)<sup>1</sup>, [Yooseung Shin](https://scholar.google.co.kr/citations?user=yCvN9Z8AAAAJ&hl=ko)<sup>1, 4</sup>, and [Heung-Il Suk](https://scholar.google.co.kr/citations?user=dl_oZLwAAAAJ&hl=ko)<sup>1</sup><br/>
> (<sup>1</sup>Department of Artificial Intelligence, Korea University) <br/>
> (<sup>2</sup>Department of Brain and Cognitive Engineering, Korea University) <br/>
> (<sup>3</sup>Samsung Electronics Company Ltd.) <br/>
> (<sup>4</sup>Heuron Company Ltd.) <br/>
> Published in IEEE TNNLS: https://doi.org/10.1109/tnnls.2025.3620853 <br/>
> 
> **Abstract:** *Single-source domain generalization (SDG) in medical image segmentation (MIS) aims to generalize a model using data from only one source domain to segment data from an unseen target domain. Despite substantial advances in SDG with data augmentation, existing methods often fail to fully consider the details and uncertain areas prevalent in MIS, leading to mis-segmentation. This paper proposes a Fourier-based semantic augmentation method called FIESTA using uncertainty guidance to enhance the fundamental goals of MIS in an SDG context by manipulating the amplitude and phase components in the frequency domain. The proposed Fourier augmentative transformer addresses semantic amplitude modulation based on meaningful angular points to induce pertinent variations and harnesses the phase spectrum to ensure structural coherence. Moreover, FIESTA employs epistemic uncertainty to fine-tune the augmentation process, improving the ability of the model to adapt to diverse augmented data and concentrate on areas with higher ambiguity. Extensive experiments across three cross-domain scenarios demonstrate that FIESTA surpasses recent state-of-the-art SDG approaches in segmentation performance and significantly contributes to boosting the applicability of the model in medical imaging modalities.*

## Overall Framework
<p align="center"><img width="90%" src="https://github.com/user-attachments/assets/9d52bbf4-9d93-4afe-afef-3f7e9eb730d9" /></p>
<p align="center"><img width="90%" src="https://github.com/user-attachments/assets/15c44327-f66a-4bf6-b0f6-21201b167689" /></p>

- To the best of our knowledge, this work is the first Fourier-based augmentation method that simultaneously manipulates amplitude and phase components using meaningful factors tailored to SDG for cross-domain MIS.
- We propose the FAT, providing an advanced augmentation strategy that combines masking and modulation techniques to transform the amplitude spectrum and applies filtering to refine the phase information to impose structural integrity.
- The FIESTA framework embraces an uncertainty-guided mutual augmentation strategy by applying UG to focus learning in the segmentation model on certain areas of high ambiguity or mis-segmented locations.
- Based on the quantitative and qualitative experimental results on various cross-domain scenarios (including cross-modality, cross-sequence, and cross-sites), we demonstrate the significant robustness and generalizability of FIESTA, which surpasses state-of-the-art SDG methods.

## Comparison with State-of-the-art SDG Methods
### Quantitative Evaluations
<p align="center"><img width="85%" src="https://github.com/user-attachments/assets/8a467907-d49a-41a5-80bf-8733b6fc9af2" /></p>
<p align="center"><img width="90%" src="https://github.com/user-attachments/assets/f4901e3f-775e-49a3-9795-b91437790d86" /></p>

### Qualitative Analyses 
<p align="center"><img width="100%" src="https://github.com/user-attachments/assets/86ba83c0-7932-4b09-bd37-58d5a8fa9251" /></p>

## Visualization of Uncertainty Guidance Effects
<p align="center"><img width="80%" src="https://github.com/user-attachments/assets/f31c7804-4f54-4c35-a60f-0e2b24651813" /></p>

## Ablation Study
<p align="center"><img width="80%" src="https://github.com/user-attachments/assets/6eee8e06-c98b-4e36-a4f3-dd6730a265d6" /></p>

## Scalability Verification Using the Segment Anything Model
<p align="center"><img width="80%" src="https://github.com/user-attachments/assets/67d9319e-da3d-4562-b67b-2ad1704872b9" /></p>
<p align="center"><img width="80%" src="https://github.com/user-attachments/assets/dd03c88d-bc57-410d-9bf5-0023378c87f9" /></p>

## Data preparation

We conduct datasets preparation following [CSDG](https://github.com/cheng-01037/Causality-Medical-Image-Domain-Generalization)

<details>
  <summary>
    <b>1) Abdominal MRI</b>
  </summary>

0. Download [Combined Healthy Abdominal Organ Segmentation dataset](https://chaos.grand-challenge.org/) and put the `/MR` folder under `./data/CHAOST2/` directory

1. Converting downloaded data (T2 SPIR) to `nii` files in 3D for the ease of reading.

run `./data/abdominal/CHAOST2/s1_dcm_img_to_nii.sh` to convert dicom images to nifti files.

run `./data/abdominal/CHAOST2/png_gth_to_nii.ipynp` to convert ground truth with `png` format to nifti.

2. Pre-processing downloaded images

run `./data/abdominal/CHAOST2/s2_image_normalize.ipynb`

run `./data/abdominal/CHAOST2/s3_resize_roi_reindex.ipynb`

The processed dataset is stored in `./data/abdominal/CHAOST2/processed/`

</details>

<details>
  <summary>
    <b>1) Abdominal CT</b>
  </summary>

0. Download [Synapse Multi-atlas Abdominal Segmentation dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) and put the `/img` and `/label` folders under `./data/SABSCT/CT/` directory

1.Pre-processing downloaded images

run `./data/abdominal/SABS/s1_intensity_normalization.ipynb` to apply abdominal window.

run `./data/abdominal/SABS/s2_remove_excessive_boundary.ipynb` to remove excessive blank region. 

run `./data/abdominal/SABS/s3_resample_and_roi.ipynb` to do resampling and roi extraction.
</details>

The details for cardiac datasets will be given later.

We also provide the [processed datasets](https://drive.google.com/file/d/1WlXGt3Nffzu1bn6co-qaidHjqWH51smU/view?usp=share_link). Download and unzip the file where the folder structure should look this:

```none
SLAug
├── ...
├── data
│   ├── abdominal
│   │   ├── CHAOST2
│   │   │   ├── processed
│   │   ├── SABSCT
│   │   │   ├── processed
│   ├── cardiac
│   │   ├── processed
│   │   │   ├── bSSFP
│   │   │   ├── LGE
├── ...
```

## Citation
If you find this work useful for your research, please cite the following paper:

```
@article{oh2024fiesta,
  title={FIESTA: Fourier-Based Semantic Augmentation with Uncertainty Guidance for Enhanced Domain Generalizability in Medical Image Segmentation},
  author={Oh, Kwanseok and Jeon, Eunjin and Heo, Da-Woon and Shin, Yooseung and Suk, Heung-Il},
  journal={arXiv preprint arXiv:2406.14308},
  year={2024}
}
```

## Acknowledgements
Our codes are built upon [CSDG](https://github.com/cheng-01037/Causality-Medical-Image-Domain-Generalization) and [SLAug](https://github.com/Kaiseem/SLAug), thanks for their contribution to the community and the development of researches!

This work was supported by the Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) No. 20220-00959 ((Part 2) Few-Shot Learning of Causal Inference in Vision and Language for Decision Making) and No. 20190-00079 (Department of Artificial Intelligence (Korea University)). This study was further supported by KBRI basic research program through Korea Brain Research Institute funded by the Ministry of Science and ICT (22-BR-03-05).
