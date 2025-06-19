
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


### Execution

#### 1.imgbeddings

image feature extractor로 imgbeddings을 사용합니다. 

imgbeddings는 OpenAI 의 강력한 CLIP 모델 과 Hugging Face transforms를 사용하여 이미지에서 임베딩 벡터를 생성하는 Python 패키지입니다 

- 임베딩 생성 모델 은 ONNX INT8 양자화되어 있어 CPU에서 20~30% 더 빠르고 디스크에서 훨씬 더 작은 크기를 자랑하며, 종속성으로 PyTorch나 TensorFlow가 필요하지 않습니다!
- CLIP의 제로샷 성능 덕분에 다양한 이미지 도메인에서 작동합니다.
- 생성된 임베딩의 차원을 줄이는 데 많은 정보를 잃지 않으면 서 주성분 분석(PCA)을 사용하기 위한 유틸리티가 포함되어 있습니다 .

[참고](https://pypi.org/project/imgbeddings/)

```
main_hugging.py
```

엔터 누르면 얼굴 등록함

#### 2. MTCNN, InceptionResnetV1

얼굴인식: MTCNN

이미지 특징 추출: InceptionResnetV1

```
main_local.py
```