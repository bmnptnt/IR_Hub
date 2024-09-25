# IR_Hub
This repository is Hub of *Hub of Image Restoration models*.
Refer the instruction below, you can implement sample process of training and testing.

## Environment
- You must install a python virtual environment of version >= 3.8 
```
conda create -n <env name> python=3.8
```
- I recommend that you install the *PyTorch 1.12.1 + CUDA 11.6* version by following the instructions below.
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
