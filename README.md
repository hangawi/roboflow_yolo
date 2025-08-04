# YOLOv8 Rabbit & Penguin Object Detection Project

이 프로젝트는 최신 객체 탐지 모델인 **YOLOv8**을 사용하여 토끼와 펭귄을 탐지하는 모델을 구축하고, 이를 동영상에 적용하는 전체 과정을 담고 있습니다. 객체 탐지 추론 방식은 **로컬(Local) 환경**과 **Roboflow API** 두 가지 방법을 모두 제공하여 다양한 환경에 대응할 수 있도록 구성했습니다.

## 기술 스택

- **언어:** Python
- **딥러닝 프레임워크:** PyTorch
- **객체 탐지 모델:** YOLOv8
- **핵심 라이브러리:** Ultralytics, OpenCV, Roboflow
- **데이터셋 관리:** Roboflow

## 왜 YOLOv8을 사용하는가?

**YOLO(You Only Look Once)**는 실시간 객체 탐지를 위해 개발된 빠르고 정확한 딥러닝 모델입니다. YOLOv8은 이전 버전에 비해 **더 높은 정확도와 빠른 속도**를 자랑하며, 사용하기 쉬운 인터페이스를 제공하여 많은 개발자들에게 사랑받고 있습니다.

### 객체 탐지의 원리

1.  **학습 (Training):** 모델에게 수많은 '토끼'와 '펭귄' 이미지를 보여주며, 이미지 내에서 해당 객체가 어디에 위치하는지(Bounding Box)와 무엇인지(Class)를 학습시킵니다. 이 프로젝트에서는 Roboflow에 공개된 데이터셋을 사용합니다.
2.  **정확도 향상:** 모델은 학습을 반복(Epochs)하면서, 주어진 이미지에서 객체의 특징을 더 잘 찾아내도록 스스로를 개선합니다. 더 많은 데이터와 더 긴 학습 시간은 일반적으로 더 높은 정확도로 이어집니다.
3.  **추론 (Inference):** 학습이 완료된 모델은 이제껏 본 적 없는 새로운 이미지나 동영상 프레임이 주어졌을 때, 학습된 지식을 바탕으로 '토끼'나 '펭귄' 객체를 스스로 찾아내고 위치를 표시할 수 있게 됩니다. 이 과정을 '추론'이라고 합니다.

## Roboflow: 데이터셋 준비의 모든 것

**Roboflow**는 컴퓨터 비전 프로젝트를 위한 데이터셋을 쉽게 구축하고 관리할 수 있도록 도와주는 플랫폼입니다. 이미지에 라벨을 붙이는 작업(Annotation)부터, 데이터의 양을 늘리는 증강(Augmentation), 그리고 다양한 포맷으로 데이터셋을 내보내는 기능까지 제공합니다. 이 프로젝트에서는 Roboflow를 통해:

-   공개된 토끼와 펭귄 데이터셋을 다운로드합니다.
-   YOLOv8 학습에 필요한 형식으로 데이터셋을 구성합니다.
-   API를 통해 학습된 모델을 배포하고 추론하는 기능을 활용합니다.

## 프로젝트 구조

```
.
├── data
│   ├── rabbit_dataset
│   └── penguin_dataset
├── scripts
│   ├── rabbit
│   │   ├── download_rabbit_dataset.py
│   │   ├── run_api_inference.py
│   │   ├── run_local_inference.py
│   │   ├── run_training.py
│   └── penguin
│       ├── download_penguin_dataset.py
│       ├── run_api_inference.py
│       ├── run_local_inference.py
│       └── run_training.py
├── raw_videos
│   ├── rabbit_original.mp4
│   └── penguin_original.mp4
├── trained_models
│   ├── penguin_best_model.pt
│   └── rabbit_trained_model.pt
├── results/             # 추론 결과 동영상 저장
│   ├── rabbit_api_detected.mp4
│   ├── rabbit_local_detected.mp4
│   ├── penguin_api_detected.mp4
│   └── penguin_local_detected.mp4
└── requirements.txt
```

## 사용 방법

### 1. 필요 라이브러리 설치

프로젝트 실행에 필요한 라이브러리를 `requirements.txt` 파일을 통해 한 번에 설치합니다.

```bash
pip install -r requirements.txt
```

### 2. 데이터셋 다운로드 (선택 사항)

데이터셋이 없는 경우, Roboflow에서 제공하는 데이터셋을 다운로드할 수 있습니다. 각 스크립트는 `data` 폴더 내에 해당 동물의 데이터셋을 다운로드합니다.

```bash
# 토끼 데이터셋 다운로드
python scripts/rabbit/download_rabbit_dataset.py

# 펭귄 데이터셋 다운로드
python scripts/penguin/download_penguin_dataset.py
```

### 3. 모델 학습: 나만의 탐정 만들기 (선택 사항)

`trained_models` 폴더에 이미 학습된 모델이 제공되지만, `run_training.py` 스크립트를 실행하여 자신만의 객체 탐지 모델을 만들 수 있습니다. 이 과정은 **전이 학습(Transfer Learning)** 이라는 효율적인 기술을 사용합니다.

#### 전이 학습(Transfer Learning)이란?

마치 사람이 새로운 것을 배울 때 기존 지식을 활용하는 것처럼, 모델도 처음부터 모든 것을 학습하는 대신, 이미 수많은 이미지로 학습된 **사전 학습 모델(Pre-trained Model)**을 기반으로 새로운 지식을 더 빠르게 습득할 수 있습니다.

`run_training.py` 스크립트는 다음과 같이 작동합니다.

1.  **`yolov8n.pt` 모델 로드:** `yolov8n.pt`는 이미 수백만 개의 다양한 이미지(사람, 자동차, 동물 등)를 학습하여 세상의 기본적인 시각적 특징을 이해하고 있는 똑똑한 기본 모델입니다.
2.  **새로운 데이터셋 학습:** 이 똑똑한 기본 모델 위에 우리가 준비한 '토끼' 또는 '펭귄' 데이터셋을 추가로 학습시킵니다.
3.  **효율적인 학습:** 모델은 이미 알고 있는 시각적 특징(예: 형태, 질감, 색상)을 바탕으로 새로운 객체(토끼, 펭귄)를 훨씬 빠르고 효율적으로 학습할 수 있습니다.

**실행 방법:**

```bash
# 토끼 모델 학습 (yolov8n.pt 기반)
python scripts/rabbit/run_training.py

# 펭귄 모델 학습 (yolov8n.pt 기반)
python scripts/penguin/run_training.py
```

학습이 완료되면, `trained_models` 폴더에 `best.pt`라는 이름으로 가장 성능이 좋은 모델 가중치 파일이 저장됩니다. 이 파일이 바로 여러분만의 'rabbit/penguin detector'입니다.

### 4. 추론 실행

동영상에서 객체 탐지를 수행하는 방법은 **로컬 추론**과 **API 추론** 두 가지가 있습니다.

#### 4.1. 로컬(Local) 추론

사용자의 컴퓨터 환경에서 직접 학습된 모델을 사용하여 추론을 수행합니다. 인터넷 연결 없이도 실행 가능하며, 속도가 빠릅니다.

- **장점:**
    - **속도:** API 호출에 비해 매우 빠릅니다. 실시간 처리에 가깝게 작동할 수 있습니다.
    - **비용:** 추가적인 API 사용 비용이 발생하지 않습니다.
    - **보안:** 데이터를 외부로 전송하지 않으므로 민감한 데이터 처리에 더 안전합니다.
    - **오프라인:** 인터넷 연결 없이도 사용할 수 있습니다.

- **단점:**
    - **성능:** 사용자의 컴퓨터 사양(CPU, GPU)에 따라 처리 속도가 크게 달라집니다. 고사양의 하드웨어가 없으면 속도가 느릴 수 있습니다.
    - **설치:** `PyTorch`, `CUDA` 등 딥러닝 라이브러리 설치 과정이 복잡할 수 있습니다.

**실행 방법:**

```bash
# 토끼 영상 로컬 추론
python scripts/rabbit/run_local_inference.py

# 펭귄 영상 로컬 추론
python scripts/penguin/run_local_inference.py
```

#### 4.2. Roboflow API 추론

Roboflow에 호스팅된 모델을 사용하여 API를 통해 추론을 수행합니다. 각 프레임을 이미지로 저장하여 API로 전송하고, 결과를 받아 동영상에 표시합니다.

- **장점:**
    - **간편함:** 복잡한 라이브러리 설치 없이 API 키만 있으면 사용할 수 있습니다.
    - **일관된 성능:** 사용자의 컴퓨터 사양에 관계없이 Roboflow 서버에서 추론을 수행하므로 일관된 성능을 기대할 수 있습니다.

- **단점:**
    - **속도:** 각 프레임을 네트워크를 통해 전송해야 하므로 매우 느립니다. 실시간 처리에 부적합합니다.
    - **비용:** Roboflow의 API 사용량 정책에 따라 비용이 발생할 수 있습니다.
    - **인터넷 필수:** 반드시 인터넷 연결이 필요합니다.
    - **보안:** 이미지를 외부 서버로 전송해야 하므로 보안에 민감한 데이터에는 적합하지 않을 수 있습니다.

**실행 방법:**

`run_api_inference.py` 스크립트 내의 `ROBOFLOW_API_KEY`를 자신의 키로 변경해야 합니다.

```bash
# 토끼 영상 API 추론
python scripts/rabbit/run_api_inference.py

# 펭귄 영상 API 추론
python scripts/penguin/run_api_inference.py
```