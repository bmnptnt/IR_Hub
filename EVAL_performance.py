import cv2
import numpy as np
from glob import glob
import os
import math
import argparse
from utils.util_calculate_psnr_ssim import calculate_psnr,calculate_ssim
import lpips
import torch

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_lq', type=str, default='./results/mct_x4',
                        help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default='testsets/SR/HR/Set5/',
                        help='input ground-truth test image folder')

    args = parser.parse_args()

    loss_fn = lpips.LPIPS(net='alex', version='0.1')
    if torch.cuda.is_available():
        loss_fn.cuda()

    psnr = 0.0
    ssim = 0.0
    dists = []

    for i, gt_img in enumerate(glob(f"{args.folder_gt}/*")):
        img_name = os.path.basename(gt_img)
        lq_img = f"{args.folder_lq}/{os.path.splitext(img_name)[0]}_mct{os.path.splitext(img_name)[1]}"

        gt = cv2.imread(gt_img, cv2.IMREAD_COLOR)
        lq = cv2.imread(lq_img, cv2.IMREAD_COLOR)

        gt_h, gt_w, _ = gt.shape
        lq_h, lq_w, _ = lq.shape

        if lq_h < gt_h:
            gt=gt[:lq_h,:,:]
        elif lq_h > gt_h:
            lq=lq[:gt_h,:,:]
        if lq_w < gt_w:
            gt=gt[:,:lq_w,:]
        elif lq_w > gt_w:
            lq=lq[:,:gt_w,:]

        gt_lpips = lpips.im2tensor(gt)
        lq_lpips = lpips.im2tensor(lq)

        gt = gt.astype(np.float32) / 255.
        gt = (gt * 255.0).round().astype(np.uint8)

        psnr += calculate_psnr(gt, lq)
        ssim += calculate_ssim(gt, lq)

        if torch.cuda.is_available():
            gt_lpips = gt_lpips.cuda()
            lq_lpips = lq_lpips.cuda()

        dist = loss_fn.forward(gt_lpips, lq_lpips)
        dists.append(dist.item())

    avg_dist = np.mean(np.array(dists))
    stderr_dist = np.std(np.array(dists)) / np.sqrt(len(dists))

    print(f"Average PSNR : {psnr / len(glob(f'{args.folder_gt}/*'))}")
    print(f"Average SSIM : {ssim / len(glob(f'{args.folder_gt}/*'))}")
    print(f"Average LPIPS Distance : {avg_dist:.5f} (+/- {stderr_dist:.5f})")