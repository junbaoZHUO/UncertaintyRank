import os
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_WA_C1")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_WA_C2")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_WA_C3")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_WA_C4")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_WA_C5")

os.system("sed -i '8s/webcam/dslr/' scripts/train_mdd10.sh")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_DA_C1")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_DA_C2")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_DA_C3")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_DA_C4")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_DA_C5")

os.system("sed -i '9s/amazon/webcam/' scripts/train_mdd10.sh")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_DW_C1")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_DW_C2")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_DW_C3")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_DW_C4")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_DW_C5")

os.system("sed -i '8s/dslr/amazon/' scripts/train_mdd10.sh")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_AW_C1")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_AW_C2")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_AW_C3")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_AW_C4")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_AW_C5")

os.system("sed -i '9s/webcam/dslr/' scripts/train_mdd10.sh")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_AD_C1")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_AD_C2")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_AD_C3")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_AD_C4")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_AD_C5")

os.system("sed -i '8s/amazon/webcam/' scripts/train_mdd10.sh")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_WD_C1")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_WD_C2")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_WD_C3")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_WD_C4")
os.system("sh scripts/train_mdd10.sh > REBUTTAL_MDDW5/CLIP_IDEAL_WD_C5")
os.system("sed -i '9s/dslr/amazon/' scripts/train_mdd10.sh")

