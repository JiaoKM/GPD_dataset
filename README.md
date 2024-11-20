# GPD_dataset

The Garment Printing Defects (GPD) dataset consists of 36 categories of print images, including 1510 images for training and 2531 images for testing. The training set only contains defect-free images, with approximately 40 images per print category, while the test set includes both defective and defect-free images. Among them, 19 categories were collected on-site from apparel factories, while other 17 categories were collected from online sources. The resolution of all images ranges from $600 \times 600$ to $2590 \times 2590$ pixels. We annotated the ground truth at the pixel level with labelme for every defective image.

[Paper](https://dl.acm.org/doi/abs/10.1007/978-981-97-8493-6_21)

`printing_dataset.py` is a PyTorch version of a dataset loading script.

![Example images for all 36 printing categories of the GPD dataset.](./image/all_types.png)

![Example images and ground truth of 8 defects.](./image/display.png)

---
If you use this dataset, please cite by
```
@inproceedings{jiao2024enhanced,
  title={Enhanced Anomaly Detection Using Spatial-Alignment and Multi-scale Fusion},
  author={Jiao, Keming and Yao, Xincheng and Wang, Lu and Zhang, Baozhu and Liu, Zhenyu and Zhang, Chongyang},
  booktitle={Chinese Conference on Pattern Recognition and Computer Vision (PRCV)},
  pages={294--308},
  year={2024},
  organization={Springer}
}
```
