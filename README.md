# Codes for Uncertainty Modeling for Robust Domain Adaptation Under Noisy Environments (TMM 2022)

## Framework
![framework4.jpg](https://github.com/junbaoZHUO/UncertaintyRank/blob/master/framework4.png)
The framework of our method. It follows a two-stage process: clean sample selection and robust domain adaptation. In the first stage, the correction network is designed to output classification responses and uncertainty. The correction network is trained with the uncertainty-reweighted classification loss, the UncertaintyRank loss and early regularization. Once the model is trained, the original noisy source domain samples are relabeled. The relabeled samples with smaller uncertainties are selected to train the adaptation network in the second stage. The predicted uncertainties of the selected samples are also preserved and utilized to reweight the classification loss and MDD loss for robustly training the adaptation network.</br>

## Instruction
This work addresses the unsupervised domain adaptation under noisy environment.</br>
Our method first select trusted samples of noisy source domain (Correction).</br>
Then robust domain adaptation is performed (Adaptation).</br>

## Citation
If you find this is helpful for you, please kindly cite our paper.</br>
@ARTICLE{Zhuo22TMM,</br>
> author={Zhuo, Junbao and Wang, Shuhui and Huang, Qingming},</br>
> journal={IEEE Transactions on Multimedia}, </br>
> title={Uncertainty Modeling for Robust Domain Adaptation Under Noisy Environments}, </br>
> year={2022},</br>
> volume={},</br>
> number={},</br>
> pages={1-14},</br>
> doi={10.1109/TMM.2022.3205457}</br>
}</br>
