import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
# from scipy import misc
import time
import imageio
from model.LiteSalNet_models import LiteSalNet
from utils1.data import test_dataset

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
argument = parser.add_argument('--testsize', type=int, default=256, help='testing size')
args = parser.parse_args()
opt = args

dataset_path = './datasets/'

model = LiteSalNet()
model.load_state_dict(torch.load('./models/LiteSalNet/LiteSalNet_EORSSD.pth'))

model.cuda()
model.eval()

test_datasets = ['EORSSD']

for dataset in test_datasets:
    save_path = './results/' + 'LiteSalNet-' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/test-images/'
    print(dataset)
    gt_root = dataset_path + dataset + '/test-labels/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    time_sum = 0

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda() if torch.cuda.is_available() else image

        time_start = time.time()

        x1, res, s1_sig, edg1, edg_s, s2, e2, s2_sig, e2_sig, s3, e3, s3_sig, e3_sig, s4, e4, s4_sig, e4_sig, s5, e5, s5_sig, e5_sig, sk1, sk1_sig, sk2, sk2_sig, sk3, sk3_sig, sk4, sk4_sig, sk5, sk5_sig = model(
            image)

        time_end = time.time()
        time_sum += (time_end - time_start)

        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = (res * 255).astype(np.uint8)

        imageio.imsave(os.path.join(save_path, name), res)

        if i == test_loader.size - 1:
            print('Running time {:.5f}'.format(time_sum / test_loader.size))
            print('Average speed: {:.4f} fps'.format(test_loader.size / time_sum))
