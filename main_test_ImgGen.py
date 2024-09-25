import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch

from models.network_mct import MCT
from  models.network_edsr import EDSR as edsr
from models.network_chat import SwinIR as swinir
from utils import util_calculate_psnr_ssim as util

from torchinfo import summary

os.environ["CUDA_VISIBLE_DEVICES"]="1"
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='edsr', help='mct, edsr,')
    parser.add_argument('--scale', type=int, default=2, help='scale factor: 1, 2, 3, 4, 8')
    parser.add_argument('--info', type=str, default=None, help='information of test')
    parser.add_argument('--model_path', type=str,
                        default='model_zoo/EDSR.pth')
    parser.add_argument('--folder_lq', type=str, default='testsets/LR/LRBI/Manga109/x2/',
                        help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None,
                        help='input ground-truth test image folder')


    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    print(f'loading model from {args.model_path}')

    model = define_model(args)
    # summary(model)
    model.eval()
    model = model.to(device)

    # setup folder and path
    folder, save_dir, border, window_size = setup(args)

    os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['psnr_b'] = []
    psnr, ssim, psnr_y, ssim_y, psnr_b = 0, 0, 0, 0, 0

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):

        # read image
        imgname, img_lq, img_gt = get_image_pair(args, path)  # image to HWC-BGR, float32
        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]],
                              (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            output = test(img_lq, model)
            output = output[..., :h_old * args.scale, :w_old * args.scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        cv2.imwrite(f'{save_dir}/{imgname}.jpg', output)

        # evaluate psnr/ssim/psnr_b
        if img_gt is not None:
            img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
            img_gt = img_gt[:h_old * args.scale, :w_old * args.scale, ...]  # crop gt
            img_gt = np.squeeze(img_gt)

            psnr = util.calculate_psnr(output, img_gt, border=border)
            ssim = util.calculate_ssim(output, img_gt, border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            if img_gt.ndim == 3:  # RGB image
                output_y = util.bgr2ycbcr(output.astype(np.float32) / 255.) * 255.
                img_gt_y = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
                psnr_y = util.calculate_psnr(output_y, img_gt_y, border=border)
                ssim_y = util.calculate_ssim(output_y, img_gt_y, border=border)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)

            print('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}; '
                  'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}; '
                  'PSNR_B: {:.2f} dB.'.
                  format(idx, imgname, psnr, ssim, psnr_y, ssim_y, psnr_b))
        else:
            print('Testing {:d} {}.jpg is Generated'.format(idx, imgname))

    # summarize psnr/

    if img_gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        print('\n{} \n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}'.format(save_dir, ave_psnr, ave_ssim))
        if img_gt.ndim == 3:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            print('-- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}'.format(ave_psnr_y, ave_ssim_y))



def define_model(args):
    # 001 classical image sr
    if args.model == 'mct':
        model = MCT()
        param_key_g = 'params'
    elif args.model =='edsr':
        model = edsr()
    elif args.model =='swinir':
        model = model = swinir(upscale=2, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')

    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model)

    return model


def setup(args):

    if args.info is not None : save_dir = f'results/{args.model}_{args.info}_x{args.scale}/{args.folder_lq.split("/")[4]}'
    else : save_dir = f'results/{args.model}_x{args.scale}/{args.folder_lq.split("/")[4]}'

    folder = args.folder_lq
    border = args.scale
    window_size = 8



    return folder, save_dir, border, window_size


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))
    img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    if args.folder_gt is not None:
        img_gt = cv2.imread(f'{args.folder_gt}{imgname}{imgext}', cv2.IMREAD_COLOR).astype(
            np.float32) / 255.
    else : img_gt = None


    return imgname, img_lq, img_gt


def test(img_lq, model):
    output = model(img_lq)

    return output


if __name__ == '__main__':
    main()
