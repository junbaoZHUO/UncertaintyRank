# Instruction for training Correction Network
pytorch 1.3.1<br>
torchvision 0.4.2<br>
python 3.6.10<br>
numpy 1.19.1<br>



Just run the following code for training Correction Network
```python
python train_off31.py
```

Put relabeled samples and their associated aggregated uncertainty together for training Adaptation Network
```python
python generate_file_for_adaptation.py
```
One can easily modified the codes for Office-Home and Bing-Caltech.

