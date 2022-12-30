# Codes for Uncertainty Modeling for Robust Domain Adaptation Under Noisy Environments (TMM 2022)

## Framework
![framework4.jpg](https://github.com/junbaoZHUO/UncertaintyRank/blob/master/framework4.png)
The framework of our method. It follows a two-stage process: clean sample selection and robust domain adaptation. In the first stage, the correction network is designed to output classification responses and uncertainty. The correction network is trained with the uncertainty-reweighted classification loss, the UncertaintyRank loss and early regularization. Once the model is trained, the original noisy source domain samples are relabeled. The relabeled samples with smaller uncertainties are selected to train the adaptation network in the second stage. The predicted uncertainties of the selected samples are also preserved and utilized to reweight the classification loss and MDD loss for robustly training the adaptation network.</br>


## Datasets
The noisified version of Office-31 and Office-home datasets can be founed in https://pan.baidu.com/s/1NW2Bkc65BhOar915DE2GbA (with code: e068) and https://pan.baidu.com/s/19zBXpNxbGAtVTNjwGAZShw (with code: fchl), respectively. Download the datasets and put them in your computer and change the paths of images in all *_list.txt files like office_list/amazon_list.txt into yours. The Bing-Caltech dataset can be found in https://vlg.cs.dartmouth.edu/projects/domainadapt.

## Instruction
Go to Correction, select trusted samples of noisy source domain.</br>
Go to Adaptation, perform robust domain adaptation.</br>

## Citation
If you find this is helpful for you, please kindly cite our paper.</br>
@ARTICLE{Zhuo22TMM,</br>
&nbsp; &nbsp; author={Zhuo, Junbao and Wang, Shuhui and Huang, Qingming},</br>
&nbsp; &nbsp; journal={IEEE Transactions on Multimedia}, </br>
&nbsp; &nbsp; title={Uncertainty Modeling for Robust Domain Adaptation Under Noisy Environments}, </br>
&nbsp; &nbsp; year={2022},</br>
&nbsp; &nbsp; volume={},</br>
&nbsp; &nbsp; number={},</br>
&nbsp; &nbsp; pages={1-14},</br>
&nbsp; &nbsp; doi={10.1109/TMM.2022.3205457}</br>
}</br>
