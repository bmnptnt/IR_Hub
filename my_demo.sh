#test dataset = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']

#python main_train_psnr.py --opt options/baek/train_edsr_sr_baseline.json
#python main_test_psnr.py --model edsr --scale 2 --model_path model_zoo/edsr_b2_x2.pth --folder_lq testsets/SR/LR/Set5/x2/ --folder_gt testsets/SR/HR/Set5/
#
#python main_test_psnr.py --model mct --scale 4 --model_path model_zoo/mct_df2k_x4.pth --folder_lq testsets/SR/LR/Urban100/x4/ --folder_gt testsets/SR/HR/Urban100/
#


#python TRAIN_psnr_yuv400_10bit.py --opt options/baek/train_edsr_YUVsr.json
#
#python TRAIN_psnr.py --opt options/baek/train_mdt_sr.json

python TEST.py --model edsr --scale 2 --model_path model_zoo/edsr_df2k_x2.pth --folder_lq testsets/SR/LR/Manga109/x2/ --folder_gt testsets/SR/HR/Manga109/
python TEST.py --model edsr --scale 4 --model_path model_zoo/edsr_df2k_x4.pth --folder_lq testsets/SR/LR/Manga109/x4/ --folder_gt testsets/SR/HR/Manga109/


python TEST.py --model mcsr8 --scale 2 --model_path model_zoo/mcsr8_df2k_x2.pth --folder_lq testsets/SR/LR/Manga109/x2/ --folder_gt testsets/SR/HR/Manga109/
python TEST.py --model mcsr8 --scale 4 --model_path model_zoo/mcsr8_df2k_x4.pth --folder_lq testsets/SR/LR/Manga109/x4/ --folder_gt testsets/SR/HR/Manga109/


python TEST.py --model swinir_light --scale 2 --model_path model_zoo/swinir_L_df2k_x2.pth --folder_lq testsets/SR/LR/Manga109/x2/ --folder_gt testsets/SR/HR/Manga109/
python TEST.py --model swinir_light --scale 4 --model_path model_zoo/swinir_L_df2k_x4.pth --folder_lq testsets/SR/LR/Manga109/x4/ --folder_gt testsets/SR/HR/Manga109/



python TEST.py --model mct --scale 2 --model_path model_zoo/mct_df2k_x2.pth --folder_lq testsets/SR/LR/Manga109/x2/ --folder_gt testsets/SR/HR/Manga109/
python TEST.py --model mct --scale 4 --model_path model_zoo/mct_df2k_x4.pth --folder_lq testsets/SR/LR/Manga109/x4/ --folder_gt testsets/SR/HR/Manga109/
