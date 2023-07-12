import torch.nn as nn
import torch
import os
import imageio
import argparse
from Code.material.CSTNet import CST_Net as Network
from Code.utils.dataloader import test_dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    # test dataset
    parser.add_argument('--data_path', type=str, default='/home/stu/zy/data/PGS224/test',help='Path to test data') 
    parser.add_argument('--pth_path', type=str, default='/home/stu/zy/CST-Net/Snapshots/save_weights_ablation_study_color/PGS224_pth/CST-Net-150-PGS224.pth',help='Path to weights file.`')
    parser.add_argument('--save_path', type=str, default='./Results/PGS224/',help='Path to save the predictions. if `semi-sup`, edit it to `Semi-Inf-Net`')
    opt = parser.parse_args()

    print("#" * 20, "\nStart Testing (CST-Net)\n{}\nThis code is written for CSTNet "
                    "\nPlease cite the paper if you use this code and dataset. "
                    "And any questions feel free to contact me "
                    "via E-mail (666zy666@163.com)\n----\n".format(opt), "#" * 20)

    model = Network()
    model = nn.DataParallel(model, device_ids=[0]).cuda()
    model.load_state_dict(torch.load(opt.pth_path, map_location='cuda'),strict=False)
    model.eval()

    image_root = '{}/images/'.format(opt.data_path)
    gt_root = '{}/masks/'.format(opt.data_path)
    test_loader = test_dataset(image_root, opt.testsize)
    os.makedirs(opt.save_path, exist_ok=True)

    for i in range(test_loader.size):
        image, name = test_loader.load_data()
        image = image.cuda()
        lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(image)
        res = lateral_map_2
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        imageio.imsave(opt.save_path + name, res)

    print('Test Done!')

if __name__ == "__main__":
    inference()
