# IR_Hub
영상 해상도 복원 모델들에 대한 실험을 위한 리포지토리입니다.
아래의 가이드를 참조하시어 학습 및 평가를 위한 예시 코드를 활용해보실 수 있습니다.

## Environment
- 3.8 버전 이상의 파이썬 가상환경이 필요합니다.
```
conda create -n <env name> python=3.8
```
- 아래의 코드와 같이 **PyTorch 1.12.1 버전과 CUDA 11.6버전**을 설치하는 것을 추천드립니다. 
- (만약 실험하고자 하는 모델이 다른 버전을 요구한다면, 해당 모델에 적합한 버전의 Pytorch와 CUDA를 설치하셔도 됩니다.)
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
- 그 외의 라이브러리는 아래의 코드로 설치 가능합니다. 
```
pip install -r requirements.txt
```

## Dataset
- 모델 학습을 위한 영상 데이터셋을 './trainsets/' 경로에 위치시켜주세요.
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
- 모델 평가를 위한 데이터셋을 './testsets/' 경로에 위치시켜주세요.
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
- 'TRAIN_psnr.py' 를 실행하여 영상 복원 모델을 학습시킬 수 있습니다. PSNR 관점에서의 복원 성능 향상을 위한 학습이 이루어집니다. 
```
python TRAIN_psnr.py --opt <모델 설정 파일 경로> 
```
- 아래의 예시 코드를 실행하여 EDSR 모델에 대한 학습을 진행할 수 있습니다. 
```
python TRAIN_psnr.py --opt options/train_edsr_sr_baseline.json
```
## Testing
- 'TEST.py' 를 실행하여 영상 복원 모델의 성능을 평가할 수 있습니다. 결과로서 화질 지표인 PSNR, SSIM과 추론 시간을 확인할 수 있습니다. 
```
python TEST_psnr.py --model <모델명> --scale <영상 해상도 배율> --model_path <사전 학습된 모델 경로> --folder_lq <저화질, 저해상도 영상 경로> --folder_gt <정답, 원본 영상 경로>
```
- 아래의 예시 코드를 실행하여 EDSR의 영상 해상도 x2배 업스케일링에 대한 성능을 확인할 수 있습니다. 
```
python TEST_psnr.py --model edsr --scale 2 --model_path model_zoo/edsr_b2_x2.pth --folder_lq testsets/SR/LR/Set5/x2/ --folder_gt testsets/SR/HR/Set5/
```
## Performance
- 아래의 예시 코드를 실행하여 두 영상 간의 객관적 화질 지표(PSNR, SSIM)과 주관적 화질 지표(LPIPS)를 확인할 수 있습니다.
```
python EVAL_performance.py --folder_lq <평가하기 위한 영상 경로> --folder_gt <정답, 원본 영상 경로>
```

## Acknowledgement
- [KAIR](https://github.com/cszn/KAIR)
- [LPIPS](https://github.com/richzhang/PerceptualSimilarity)
