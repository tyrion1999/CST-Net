# The function of eval_best.py is to compute a series of metrics, such as Dice Jaccard, between the generated predictor and the label.

import os
import torch.nn.functional as F
import cv2
import numpy as np
import torch

from metrics2 import evaluate

# img_path = r'/home/stu/zy/Inf-Net-master/Results/polyp/Semi-Inf-Net/Kvasir' # Path to the generated pred
img_path = r'/home/stu/zy/Inf-Net-master/Results/Ablation/skip_connection_layer2_adaptive_fusion_positional_embedding_MLP+/ISIC2018' # Path to the generated pred
# label_path = r'/home/stu/zy/Inf-Net-master/Dataset/MTNS224/test/labelcol' # Label paths in the original test dataset
label_path = r'/home/stu/zy/data/ISIC2018/test/masks' # Label paths in the original test dataset

def test_one(img_path,label_path):

    recall = 0.0
    specificity = 0.0
    precision = 0.0
    f1 = 0.0
    f2 = 0.0
    acc = 0.0
    iou_thyroid = 0.0
    iou_bg = 0.0
    miou = 0.0
    dice = 0.0
    jaccard = 0.0
    count = 0

    for filename in os.listdir(label_path): # filename：(566)_6.png
        count =count+1
        pred = cv2.imread(img_path+'/'+filename)
        # pred = np.array(pred)
        pred = torch.Tensor(pred) # (352,352,3)
        label = cv2.imread(label_path+'/'+filename)
        # If the sizes of pred and label are different, change the label to the size of pred.
        label = cv2.resize(label,(pred.size()[0],pred.size()[1]))
        label = np.array(label)
        label = torch.Tensor(label)

        Recall, Specificity, Precision, F1, F2, ACC_overall, IoU_poly, IoU_bg, IoU_mean, Dice, Jaccard = evaluate(pred,label)

        recall = recall + Recall
        specificity = specificity + Specificity
        precision = precision + Precision
        f1 = f1 + F1
        f2 = f2 + F2
        acc = acc + ACC_overall
        # iou_thyroid = iou_thyroid + IoU_Thyroid
        iou_bg = iou_bg + IoU_bg
        miou = miou + IoU_mean
        dice = dice + Dice 
        jaccard = jaccard +Jaccard
    # print('length of dataset:{}'.format(count))
    # print('recall:{:.3f}'.format(recall/count))
    # print('specificity:{:.3f}'.format(specificity/count))
    # print('precision:{:.3f}'.format(precision/count))
    # print('f1:{:.3f}'.format(f1/count))
    # print('f2:{:.3f}'.format(f2/count))
    # print('acc:{:.3f}'.format(acc/count))
    # print('iou_thyroid:{:.3f}'.format(iou_thyroid/count))
    # print('iou_bg:{:.3f}'.format(iou_bg/count))
    # print('Miou:{:.3f}'.format(miou/count))

    return dice /count, jaccard/count, acc/count, recall/count, specificity/count

def iou():

    flag_path = ''
    flag_dice = 0.0
    flag_jaccard = 0.0
    flag_acc = 0.0
    flag_sensitivity = 0.0
    flag_specificity = 0.0
    for sub_path in os.listdir(img_path): # sub_path:'(566)_6.png'
        iou = test_one(img_path,label_path)
        if iou[0] > flag_dice:
            flag_dice = iou[0]
            flag_jaccard = iou[1]
            flag_acc = iou[2]
            flag_sensitivity = iou[3]
            flag_specificity = iou[4]
            flag_path = img_path+'/'+sub_path
    return flag_dice, flag_jaccard, flag_acc, flag_sensitivity, flag_specificity, flag_path

result = iou()

print('***************')
print('Dice:{:.4f}'.format(result[0]))
print('Jaccard:{:.4f}'.format(result[1]))
print('Accuracy:{:.4f}'.format(result[2]))
print('Sensitivity:{:.4f}'.format(result[3]))
print('Specificity:{:.4f}'.format(result[4]))
print('path：{}'.format(result[5]))
print('***************')
