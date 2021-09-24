# Personalized Retrogress-Resilient Federated Learning (PRR-FL)
This repository is an official PyTorch implementation of the paper "Personalized Retrogress-Resilient Framework for Real-World Medical Federated Learning" from MICCAI 2021.

## Retrogress Challenge
<div align=center><img width="750" src=/figs/retrogress_curve.png></div>
In the scenarios of real-world medical FL, an abrupt performance drop (termed as retrogress) of existing FL methods after each server-client communication. The inter-client data heterogeneity leads to enormous discrepancy between different local models. In this case, it becomes unreasonable to average parameters of client models in element-wise at the server side and to replace the previous local models with the aggregated server model at the client side.


## Personalized Retrogress-Resilient FL Framework

<div align=center><img width="600" src=/figs/framework.png></div>

**Personalized Retrogress-Resilient FL Framework** aims to generate a personalized model with superior performance for each client. By introducing a deputy model to exchange the knowledge between the server and the client, the personalized model can keep the stable local training, without being disturbed by the communication. This framework includes two improvements at the server and clients:

**Progressive Fourier Aggregation (PFA)** aggregates the relatively low-frequency components of parameters to share the client knowledge, while preserving the individual high-frequency components. For the low-frequency mask, we progressively increase the frequency threshold of the shared component during the FL to stabilize the training.

**Deputy-Enhanced Transfer (DET)** transfers the global knowledge between the deputy model and the personalized model in Recover-Exchange-Sublimate. At the beginning of each iteration, the deputy updated with the server model has poor performance due to the retrogress problem. Thus, the deputy should be improved firstly, and then transfer the knowledge to the personalized model.

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

## Quickstart 
* Train the PRR-FL with default settings:
```python
python ./main.py --theme prrfl
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
