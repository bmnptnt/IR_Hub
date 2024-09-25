import cv2
import numpy as np
from glob import glob
import os
import math
import argparse
from utils.util_calculate_psnr_ssim import calculate_psnr,calculate_ssim

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_lq', type=str, default='testsets/LR/LRBI/Manga109/x2/',
                        help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default='testsets/HR/Manga109/x2/',
                        help='input ground-truth test image folder')

    args = parser.parse_args()

    psnr = 0.0
    ssim = 0.0

    for i, gt_img in enumerate(glob(f"{args.folder_gt}/*")):
        img_name = os.path.basename(gt_img)
        lq_img = f"{args.folder_lq}/{os.path.splitext(img_name)[0]}{os.path.splitext(img_name)[1]}"
        gt = cv2.imread(gt_img, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        gt = (gt * 255.0).round().astype(np.uint8)
        lq = cv2.imread(lq_img, cv2.IMREAD_COLOR)
        psnr += calculate_psnr(gt, lq)
        ssim += calculate_ssim(gt, lq)

    print(f"PSNR : {psnr / len(glob(f'{args.folder_gt}/*'))}")
    print(f"SSIM : {ssim / len(glob(f'{args.folder_gt}/*'))}")