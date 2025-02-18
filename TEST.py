import argparse
import cv2
import glob
import numpy as np
from collections import OrderedDict
import os
import torch

from utils import util_calculate_psnr_ssim as util
from torchinfo import summary
import time
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='edsr', help='mct, edsr,')
    parser.add_argument('--scale', type=int, default=2, help='scale factor: 1, 2, 3, 4, 8')
    parser.add_argument('--channel', type=int, default=3, help='number of channel: 1, 2, 3, 4')
    parser.add_argument('--bit', type=int, default=8, help='input bit depth: 8, 16')
    parser.add_argument('--model_path', type=str,
                        default='model_zoo/EDSR.pth')
    parser.add_argument('--folder_lq', type=str, default='testsets/LR/LRBI/Manga109/x2/',
                        help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default='testsets/HR/Manga109/x2/',
                        help='input ground-truth test image folder')

    parser.add_argument('--info', type=str, default=None, help='information of network')


    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    print(f'loading model from {args.model_path}')

    model = define_model(args)
    summary(model)
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

    total_t=0.0
    first_t=0.0
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

            start_t=time.time()
            output = test(img_lq, model)
            end_t=time.time()-start_t
            if idx==0 : first_t=end_t
            total_t+=end_t
            output = output[..., :h_old * args.scale, :w_old * args.scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR

        if args.bit == 16:
            output = (output * 65535.0).round().astype(np.uint16)  # float32 to uint16
        else:
            output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        cv2.imwrite(f'{save_dir}/{imgname}_{args.model}.png', output)

        # evaluate psnr/ssim/psnr_b
        if img_gt is not None:
            if args.bit ==16:
                img_gt = (img_gt * 65535.0).round().astype(np.uint16)  # float32 to uint8
            else :
                img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8

            img_gt = img_gt[:h_old * args.scale, :w_old * args.scale, ...]  # crop gt
            img_gt = np.squeeze(img_gt)
            if args.bit ==16:
                psnr = util.calculate_psnr16(output, img_gt, border=border)
                ssim = util.calculate_ssim16(output, img_gt, border=border)
            else :
                psnr = util.calculate_psnr(output, img_gt, border=border)
                ssim = util.calculate_ssim(output, img_gt, border=border)

            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            if img_gt.ndim == 3:  # RGB image
                if args.bit == 16:
                    output_y = util.bgr2ycbcr(output.astype(np.float32) / 65535.) * 65535.
                    img_gt_y = util.bgr2ycbcr(img_gt.astype(np.float32) / 65535.) * 65535.
                else :
                    output_y = util.bgr2ycbcr(output.astype(np.float32) / 255.) * 255.
                    img_gt_y = util.bgr2ycbcr(img_gt.astype(np.float32) / 255.) * 255.
                if args.bit == 16:
                    psnr_y = util.calculate_psnr16(output_y, img_gt_y, border=border)
                    ssim_y = util.calculate_ssim16(output_y, img_gt_y, border=border)
                else :
                    psnr_y = util.calculate_psnr(output_y, img_gt_y, border=border)
                    ssim_y = util.calculate_ssim(output_y, img_gt_y, border=border)

                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)

            print('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f}; '
                  'PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}; '
                  'PSNR_B: {:.2f} dB. '
                  'Elasped time : {}'.
                  format(idx, imgname, psnr, ssim, psnr_y, ssim_y, psnr_b, end_t))
        else:
            print('Testing {:d} {:20s} Elasped time : {}'.format(idx, imgname, end_t))

    # summarize psnr/ssim
    if img_gt is not None:

        avg_t=(total_t-first_t)/(len(glob.glob(os.path.join(folder, '*')))-1)

        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['psnr'])
        print(f"length of data : { len(test_results['psnr'])}")
        print('\n{} \n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f}\n Total Time : {} sec Average Time : {} ms'.format(save_dir, ave_psnr, ave_ssim, total_t,avg_t*1000.0))
        if img_gt.ndim == 3:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            print('-- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}'.format(ave_psnr_y, ave_ssim_y))



def define_model(args):
    # 001 classical image sr
    if args.model == 'mct':
        from models.network_mct import MCT as net
        model = net(in_channels=args.channel,
                    dim=[48, 48, 24, 24, 24, 12, 12, 8, 4],
                    scale=args.scale,
                    heads=[8, 8, 6, 6, 6, 4, 4, 2, 1],
                    ffn_expansion_factor=3,
                    bias=False,
                    LayerNorm_type='WithBias')

    elif args.model =='edsr':
        from models.network_edsr_chAdapt import EDSR as net
        model = net(in_chans=args.channel,scale=args.scale,n_feats=64,n_resblocks=16)
    elif args.model =='swinir_light':
        from models.network_swinir import SwinIR as net
        model = net(upscale=args.scale, in_chans=args.channel, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')

    elif args.model =='swinir':
        from models.network_swinir import SwinIR as net
        model = net(upscale=args.scale, in_chans=args.channel, img_size=48, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')

    elif args.model == 'swinir_nnpf':
        from models.network_swinir import SwinIR as net
        model = net(upscale=args.scale, in_chans=args.channel, img_size=128, window_size=8,
                       img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                       mlp_ratio=2, upsampler=None, resi_connection='1conv')

    elif args.model =='rcan':
        from models.network_rcan import RCAN as net
        model = net(scale=args.scale,
                   n_resgroups=3,
                   n_resblocks=6,
                   n_feats=64)
    elif args.model =='mcsr8':
        from models.network_mcsr8_2 import MCSR8 as net
        model = net(scale=args.scale,
                   in_chans=args.channel,
                   n_feats=[48, 48, 24, 24, 24, 12, 12, 8])
    elif args.model =='rdsr':
        from models.network_rdsr import RDSR as net
        model = net(upscale=args.scale,
                   in_chans=args.channel,
                   feature_channels=64)
    elif args.model =='han':
        from models.network_han import HAN as net
        model = net(scale=args.scale,
                   n_resgroups=3,
                   n_resblocks=6,
                   n_feats=64)
    elif args.model =='erbn':
        from models.network_erbn import ERBN as net
        model = net(scale=args.scale,
                   in_chans=args.channel,
                   n_feats= 64,
                   n_blocks=9)
    elif args.model =='rdsr_le':
        from models.network_rdsr_le import RDSR_LE as net
        model = net(upscale=args.scale,
                    in_channels=args.channel,
                    out_channels=3,
                   feature_channels=52)
    elif args.model =='rlfn':
        from models.network_rlfn import RLFN as net
        model = net(upscale=args.scale,
                    in_channels=args.channel,
                    out_channels=3,
                   feature_channels=52)
    elif args.model =='mcsr8_bi':
        from models.network_mcsr8_2 import MCSR8 as net
        model = net(scale=args.scale,
                   in_chans=args.channel,
                   n_feats=[48, 48, 24, 24, 24, 12, 12, 8])
    elif args.model == 'hat':
        from models.network_hat import HAT as net
        model = net(upscale=2,
                   in_chans=args.channel,
                   img_size=64,
                   window_size=16,
                   img_range=1.0,
                   depths=[6, 6, 6, 6, 6, 6],
                   embed_dim=144,
                   num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2,
                   upsampler="pixelshuffle",
                   resi_connection="1conv"  ,
                   compress_ratio=24,
                   squeeze_factor=24,
                   conv_scale=0.01)

    elif args.model == 'chat':
        from models.network_chat import CHAT as net
        model = net(upscale=2,
                   in_chans=args.channel,
                   img_size=64,
                   window_size=8,
                   img_range=1.0,
                   depths=[6, 6, 6, 6, 6, 6],
                   embed_dim=108,
                   num_heads=[6, 6, 6, 6, 6, 6],
                   mlp_ratio=2,
                   upsampler="pixelshuffledirect",
                   resi_connection="1conv")

    elif args.model == 'dct':
        from models.network_dct import DCT as net
        model=net(in_channels=args.channel,
                    dim=16,
                    scale=args.scale,
                    heads=4,
                    ffn_expansion_factor=2.66,
                    bias=0,
                    LayerNorm_type="WithBias")

    elif args.model == 'rct':
        from models.network_rct import RCT as net
        model = net(in_chans=args.channel,
                    feature_channels=56,
                    upscale=args.scale,
                    num_heads=8,
                    ffn_factor=3,
                    bias=False,
                    norm_type="WithBias")

    elif args.model == 'adsr3':
        from models.network_adsr_3 import ADSR as net
        model = net(upscale=2,
                   in_chans=args.channel,
                   feature_channels=72,
                   norm_type="WithBias",
                   num_heads=8,
                   ffn_expansion_factor=3.0
                   )

    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model)

    return model


def setup(args):

    if args.info is not None : save_dir = f'results/{args.model}_x{args.scale}_{args.info}'
    else : save_dir = f'results/{args.model}_x{args.scale}'
    folder = args.folder_gt
    # folder = args.folder_lq
    border = args.scale
    window_size = 8



    return folder, save_dir, border, window_size

# def get_image_pair(args, path):
#     (imgname, imgext) = os.path.splitext(os.path.basename(path))
#     img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
#     print(f'{args.folder_gt}{imgname.split("_")[0]}{imgext}')
#     img_gt = cv2.imread(f'{args.folder_gt}{imgname.split("_")[0]}{imgext}', cv2.IMREAD_COLOR).astype(
#         np.float32) / 255.
#
#     return imgname.split("_")[0], img_lq, img_gt


def get_image_pair(args, path):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    if args.bit == 16:
        img_gt = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.
        img_lq = cv2.imread(f'{args.folder_lq}{imgname}{imgext}', cv2.IMREAD_UNCHANGED).astype(
            np.float32) / 65535.
    else :
        img_gt = cv2.imread(path).astype(np.float32) / 255.
        img_lq = cv2.imread(f'{args.folder_lq}{imgname}{imgext}').astype(
            np.float32) / 255.

    if args.channel == 1:
        img_gt=np.expand_dims(img_gt, axis=-1)
        img_lq=np.expand_dims(img_lq, axis=-1)

    return imgname, img_lq, img_gt


def test(img_lq, model):
    output = model(img_lq)

    return output


if __name__ == '__main__':
    main()
