# Personalized Retrogress-Resilient Federated Learning (PRR-FL)
This repository is an official PyTorch implementation of the paper "Personalized Retrogress-Resilient Framework for Real-World Medical Federated Learning" from MICCAI 2021.

## Retrogress Challenge
<div align=center><img width="750" src=/figs/retrogress_curve.png></div>



## Personalized Retrogress-Resilient FL Framework

<div align=center><img width="600" src=/figs/framework.png></div>

**Personalized Retrogress-Resilient FL Framework** .


### Download
The dermoscopic FL dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1N4bNcy09nizkEi___venM0su0hf23jO_?usp=sharing). Put the downloaded ```clientA```, ```clientB```, ```clientC``` and ```clientD``` subfolders in a newly-built folder ```./data/```.

## Dependencies
* Python 3.7
* PyTorch >= 1.7.0
* numpy 1.19.4
* scikit-learn 0.24.2
* scipy 1.6.2
* albumentations 0.5.2

## Code
Clone this repository into any place you want.
```bash
git clone https://github.com/CityU-AIM-Group/PRR-FL.git
cd PRR-FL
mkdir experiment; mkdir data
```


## Cite
If you find our work useful in your research or publication, please cite our work:
```
@inproceedings{2021personalized,
  title={Personalized Retrogress-Resilient Framework for Real-World Medical Federated Learning},
  author={Chen, Zhen and Zhu, Meilu and Yang, Chen and Yuan, Yixuan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2021},
  organization={Springer}
}
```

## Acknowledgements
* Federated learning framework derived from [FedBN](https://github.com/med-air/FedBN).
