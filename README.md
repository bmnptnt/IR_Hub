# IR_Hub
This repository is **Hub of Image Restoration models**.
Refer the instruction below, you can implement sample process of training and testing.

*※ 한국어 가이드는 [README_kr.md](https://github.com/bmnptnt/IR_Hub/blob/main/README_kr.md) 파일을 참조하시기 바랍니다.*

## Environment
- You must install a python virtual environment of version >= 3.8 
```
conda create -n <env name> python=3.8
```
- I recommend that you install the **PyTorch 1.12.1 + CUDA 11.6** version by following the instructions below.
- (If your model require another version of pytorch, you can choose suitable version.)
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
- Other libraries can be installed with the following command.
```
pip install -r requirements.txt
```

## Dataset
- Place your training dataset in the './trainsets/' path.
```
├── trainsets/
│    └── SR/
│         ├── HR/
│         │    └── DIV2K_train_HR/
│         │        ├── 0001.png
│         │        ├── ...
│         │            
│         └── LR/            
│              └── DIV2K_train_LR_bicubic/       
│                   ├── x2/  
│                   │     ├── 0001.png
│                   │     ├── ... 
│                   │ 
│                   ├── x3/
│                   └── x4/
│
```
- Place your testing dataset in the './testsets/' path.
```
├── testsets/
│    └── SR/
│         ├── HR/
│         │    ├── B100/
│         │    │   ├── 3096.png
│         │    │   ├── ...
│         │    │
│         │    └── Set5/ 
│         │
│         └── LR/            
│              ├── B100/       
│              │    ├── x2/  
│              │    │     ├── 3096.png
│              │    │     ├── ... 
│              │    │ 
│              │    ├── x3/
│              │    └── x4/
│              │
│              └── Set5/
│  
```

## Training
- By running 'main_train_psnr.py', You can train image restoration model to improve PSNR.
```
python main_train_psnr.py --opt <Path of option file about model> 
```
- By following example commands, You can train the EDSR newtork.
```
python main_train_psnr.py --opt options/train_edsr_sr_baseline.json
```
## Testing
- By running 'main_test_psnr.py', You can test image restoration model. Result of testing, You can get the PSNR, SSIM score and inference time.
```
python main_test_psnr.py --model <name of model> --scale <resolution scalig factor> --model_path <path of pretrained model> --folder_lq <path of low quality images> --folder_gt <path of grount truth images>
```
- By following example commands, You can test the EDSR newtork for x2 image super resolution.
```
python main_test_psnr.py --model edsr --scale 2 --model_path model_zoo/edsr_b2_x2.pth --folder_lq testsets/SR/LR/Set5/x2/ --folder_gt testsets/SR/HR/Set5/
```

## Performance
- By following commands, You can check objective quality(PSNR,SSIM) and subjective quality(LPIPS) between generated iamges and ground truth images.
```
python check_performance.py --folder_lq <path of generated image> --folder_gt <path of ground truth iamge>
```

## Acknowledgement
- [KAIR](https://github.com/cszn/KAIR)
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity)
