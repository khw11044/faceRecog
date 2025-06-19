
# Image Retrieval를 이용한 Face Classification 

## Approach 설명 

one-shot 방법의 Image Retrieval를 통해 사람의 얼굴을 구별하는 방법입니다.

새로운 사람의 얼굴에서 feature extraction을 통해 vector를 뽑아내면, 이를 db에 저장합니다.

사람의 얼굴이 인식되어 사람 얼굴 bounding box를 crop하고 vector를 추출하여 db에서 retrieval를 진행합니다. 

유사도가 가장 가까운 db vector의 라벨과 매칭하여 얼굴을 구분할 수 있습니다.



## Getting Started

### Setting Virtual Environments

```bash
conda create -n face python=3.11 -y

conda activate face 
```

### Installation

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

```
pip install -r requirements.txt
```


엔터 누르면 얼굴 등록함
