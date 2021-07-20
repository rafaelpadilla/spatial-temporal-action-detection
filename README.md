# spatial-temporal-action-detection

## Getting Started

###  1. Setup Environment

#### 1.1 Create Virtual Environment

Create Virtual (Windows) Environment:

```shell script
python -m venv env
.\env\Scripts\activate
```

Create Virtual (Linux/Mac) Environment:

```shell script
python -m venv env
source env/bin/activate
```

#### 1.2 Install packages
```shell script
pip install -r requirements.txt
```

```shell script
# install torch with cuda
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```