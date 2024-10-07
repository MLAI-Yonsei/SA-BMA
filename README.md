# Flat Posterior Does Matter For Bayesian Model Averaging

>This repository contains PyTorch implemenations of "Flat Posterior Does Matter For Bayesian Model Averaging".
>> [Sungjun Lim](https://sungjun98.github.io/), [Jeyoon Yeom](https://www.linkedin.com/in/jeyoon-yeom-31107b2a3/), [Sooyon Kim](https://github.com/sooyonee), [Hoyoon Byun](https://drive.google.com/file/d/14YhY6kEkBV-r9F3zC3Fu3jprCQ0MzmH7/view), [Jinho Kang](https://bubble3jh.github.io/), [Yohan Jung](https://e2ee22.github.io/), [Jiyoung Jung](https://rcv.uos.ac.kr/), [Kyungwoo Song](https://mlai.yonsei.ac.kr/)

------
------

</br>

## Abstract
![flatness_description](https://github.com/user-attachments/assets/4e538693-2810-4c6f-8ced-d447acb1ce61)

> * Sharpness-Aware Bayesian Model Averaging (SA-BMA) is a novel optimizer that seeks flat posteriors by calculating divergence in the parameter space.
> * Bayesian Transfer Learning scheme efficiently leverages the  pre-trained model.

------
------

</br>

## Setup

```
git clone https://github.com/SungJun98/SA-BMA.git
cd SA-BMA

# Create and activate a conda environment
conda create -y -n sabma python=3.9.13
conda activate sabma

# install packages
pip install -r requirements.txt
```

------
------

</br>

## Data preparation
> * **CIFAR10, CIFAR100**
> We simply use *torchvision.datasets* for these datsets.

> * **OxfordPets, Flowers102, EuroSAT, UCF101, and ImageNet**
> To prepare above 5 datasets (adopted by [CoOp](https://github.com/KaiyangZhou/CoOp)), please follow the instruction from https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md
> We use the same few-shot split of **CoOp** for above 5 datasets.

> * **CIFAR10C, CIFAR100C**
> You can download CIFAR1C in https://zenodo.org/records/2535967
> You can download CIFAR100C in https://zenodo.org/records/3555552

> * **ImageNet-V2, ImageNet-R, ImageNet-A, ImageNet-Sketch**
> To prepared 4 variants of ImageNet, please follow the instructino from https://github.com/mlfoundations/wise-ft/blob/master/datasets.md

---------
---------

</br>

## Run
We explain the code snippet for each method in the case of training CIFAR-100 10-shot with a pre-trained ResNet-18.

The `run_baseline.py` allows to run all baselines, except MCMC and E-MCMC.

The `run_mcmc.py` allows to run MCMC and E-MCMC.

### SGD
```
run_baseline.py --dataset=CIFAR100 --data_path=<DATA_PATH> --dat_per_cls=10 \
--method=dnn --model=resnet18 --pre_trained --epochs=100 \
--optim=sgd --lr_init=5e-3 --wd=5e-4 
```

* ```--pre-trained``` &mdash; use pre-trained model.
* ```EPOCHS``` &mdash; number of epochs
* ```LR_INIT``` &mdash; initial learning rate
* ```WD``` &mdash; weight decay



### SAM [(Foret et al. 2021)](https://arxiv.org/pdf/2010.01412) [[code base]](https://github.com/davda54/sam)
```
run_baseline.py --dataset=CIFAR100 --data_path=<DATA_PATH> --dat_per_cls=10 \
--method=dnn --model=resnet18 --pre_trained  --epochs=100 \
--optim=sam --lr_init=5e-3 --wd=5e-4 --rho=5e-2
```
* ```RHO``` &mdash; neighborhood size of perturbation.

### FSAM [(Kim et al. 2022)](https://arxiv.org/pdf/2206.04920) ~~[code base]~~
```
run_baseline.py --dataset=CIFAR100 --data_path=<DATA_PATH> --dat_per_cls=10 \
--method=dnn --model=resnet18 --pre_trained  --epochs=100 \
--optim=fsam --lr_init=5e-3 --wd=5e-4 --rho=5e-2 --eta=1e-1
```

* ```RHO``` &mdash; neighborhood size of perturbation.
* ```ETA``` &mdash; scaling hyperparameter for the inverse of diagonal fisher.

### bSAM [(Möllenhoff et al. 2023)](https://arxiv.org/pdf/2210.01620) [[code base]](https://github.com/team-approx-bayes/bayesian-sam)
```
run_baseline.py --dataset=CIFAR100 --data_path=<DATA_PATH> --dat_per_cls=10 \
--method=dnn --model=resnet18 --pre_trained  --epochs=150 \
--optim=bsam --lr_init=5e-3 --wd=5e-4 --rho=5e-2 \
--damping=1e-4 --s_init=1e-2 --noise_scale=1e-2
```
* ```RHO``` &mdash; neighborhood size of perturbation. 
* ```DAMPING``` &mdash; stabilizes the method by adding constant when updating variance estimate.
* ```S_INIT``` &mdash; scale of variance initialization.
* ```NOISE_SCALE``` &mdash; additional hyperparameter to mitigate the gap between training from scratch and few-shot fine-tuning on the pre-trained model.

### MOPED [(Krishnan et al. 2020)](https://ojs.aaai.org/index.php/AAAI/article/view/5875) [[code base]](https://github.com/IntelLabs/bayesian-torch)
```
run_baseline.py --dataset=CIFAR100 --data_path=<DATA_PATH> --dat_per_cls=10 \
--method=vi --model=resnet18 --pre_trained --epochs=100 \
--optim=sgd --lr_init=5e-3 --wd=5e-4 \
--moped_delta=2e-1 --kl_beta=1e-1
```
* ```MOPED_DELTA``` &mdash; adjusts how much to incorporate pre-trained weights.
* ```KL_BETA``` &mdash; adjusts KLD term in VI objective function.

### MCMC (SGLD) [(Welling et al. 2011)](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf) [[code base]](https://github.com/lblaoke/EMCMC) 
```
run_mcmc.py --dataset=CIFAR100 --data_path=<DATA_PATH> --dat_per_cls=10 \
--method=mcmc --model=resnet18 --pre_trained --epochs=150 \
--optim=sgd --lr_init=5e-2 --lr_end=1e-4 --wd=5e-4 \
--n_cycle=4 --temp=1e-2
```

* ```N_CYCLE``` &mdash; number of cycle.
* ```TEMP``` &mdash; adjusts the step size of weight update.

### E-MCMC (SGLD) [(Li et al. 2023)](https://arxiv.org/pdf/2310.05401) [[code base]](https://github.com/lblaoke/EMCMC) 
```
run_mcmc.py --dataset=CIFAR100 --data_path=<DATA_PATH> --dat_per_cls=10 \
--method=mcmc --model=resnet18 --pre_trained --epochs=150 \
--optim=sgd --lr_init=5e-2 --lr_end=1e-4 --wd=5e-4 \
--n_cycle=4 --temp=1e-2 --eta=1e-2
```
* ```N_CYCLE``` &mdash; number of cycle.
* ```TEMP``` &mdash; adjusts the step size of weight update.
* ```ETA``` &mdash; handles flatness.

### SWAG [(Maddox et al. 2019)](https://arxiv.org/pdf/1902.02476) [[code base]](https://github.com/wjmaddox/swa_gaussian) 
```
run_baseline.py --dataset=CIFAR100 --data_path=<DATA_PATH> --dat_per_cls=10 \
--method=swag --model=resnet18 --pre_trained --epochs=150 \
--optim=sgd --lr_init=5e-3 --wd=5e-4 \
--swa_start=76 --max_num_models=5
```
* ```SWA_START``` &mdash; the epoch to start SWA.
* ```MAX_NUM_MODELS``` &mdash; low rank for covariance.

### F-SWAG [(Nguyen et al. 2023)](https://arxiv.org/pdf/2302.02713) ~~[code base]~~
```
run_baseline.py --dataset=CIFAR100 --data_path=<DATA_PATH> --dat_per_cls=10 \
--method=swag --model=resnet18 --pre_trained --epochs=150 \
--optim=sgd --lr_init=5e-3 --wd=5e-4 \
--swa_start=76 --max_num_models=5
```
* ```RHO``` &mdash; neighborhood size of perturbation. 
* ```SWA_START``` &mdash; the epoch to start SWA.
* ```MAX_NUM_MODELS``` &mdash; low rank for covariance.

</br>

### SA-BMA

The `run_baseline.py` make pre-trained DNN into BNN prior with specific arguments.
We give example of SWAG for this step.

```
run_baseline.py --dataset=CIFAR100 --data_path=<DATA_PATH> --dat_per_cls=10 \
--method=swag --model=resnet18 --pre_trained --epochs=150 \
--optim=sgd --lr_init=5e-3 --wd=5e-4 \
--swa_start=76 --max_num_models=5
```

The `run_sabma.py` allows to run SA-BMA with the converted prior.

```
run_sabma.py --dataset=CIFAR100 --data_path=<DATA_PATH> \
--model=resnet18 --pre_trained --epochs=150 \
--optim=sabma --lr_init=5e-2 --wd=5e-4 --rho=5e-1 \
--prior_path=<PRIOR_PATH> --alpha=1e-5 --low_rank=-1
```

* ```RHO``` &mdash; neighborhood size of perturbation. 
* ```PRIOR_PATH``` &mdash; path of prior model. 
* ```ALPHA``` &mdash; scales the variance of last layer.
* ```LOW_RANK``` &mdash; low-rank of covariance. `-1` denotes using the rank of prior.

---------
---------

</br>

## Contact
For any questions, discussions, and proposals, please contact to `lsj9862@yonsei.ac.kr` or `kyungwoo.song@gmail.com`


---------
---------

</br>

## Citation
If you use our code in your research, please kindly consider citing:
```bibtex
@article{lim2024flat,
  title={Flat Posterior Does Matter For Bayesian Transfer Learning},
  author={Lim, Sungjun and Yeom, Jeyoon and Kim, Sooyon and Byun, Hoyoon and Kang, Jinho and Jung, Yohan and Jung, Jiyoung and Song, Kyungwoo},
  journal={arXiv preprint arXiv:2406.15664},
  year={2024}
}
```

---------
---------

</br>

## Acknowledgements
We refered the code from repositories of [SWAG](https://github.com/wjmaddox/swa_gaussian) (Maddox et al.), [BlackVIP](https://github.com/changdaeoh/BlackVIP) (Oh et al.), [WiSE-FT](https://github.com/mlfoundations/wise-ft) (Wortsman et al.), [E-MCMC](https://github.com/lblaoke/EMCMC) (Li et al.), [SAM](https://github.com/davda54/sam) (Foret et al.), [MOPED](https://github.com/IntelLabs/bayesian-torch) (Krishnan et al.) , and [bSAM](https://github.com/team-approx-bayes/bayesian-sam) (Moellenhoff et al.).
We appreciate the authors for sharing their code.
