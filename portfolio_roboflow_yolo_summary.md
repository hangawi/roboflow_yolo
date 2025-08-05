# Real-Time Object Detection with YOLOv8 – Rabbit & Penguin Tracker (Roboflow API vs Local Inference Comparative Project)

이 프로젝트는 최신 객체 탐지 모델인 **YOLOv8**을 활용하여 토끼와 펭귄을 실시간으로 탐지하는 모델을 구축하고, 이를 동영상에 적용하는 과정을 다룹니다. 특히, 객체 탐지 추론 방식을 **로컬(Local) 환경**과 **Roboflow API** 두 가지로 구현하여 각 방식의 성능과 활용성을 비교 분석한 것이 특징입니다.

## 기술 스택

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Roboflow](https://img.shields.io/badge/Roboflow-4299E1?style=for-the-badge&logo=roboflow&logoColor=white)

-  **객체 탐지 모델:** YOLOv8
-  **핵심 라이브러리:** Ultralytics

## 왜 YOLOv8을 사용하는가?

**YOLO(You Only Look Once)**는 실시간 객체 탐지를 위해 개발된 빠르고 정확한 딥러닝 모델입니다. YOLOv8은 이전 버전에 비해 더 높은 정확도와 빠른 속도를 자랑하며, 사용하기 쉬운 인터페이스를 제공하여 많은 개발자들에게 사랑받고 있습니다.

### 객체 탐지의 원리

1. **학습 (Training):** 모델에게 수많은 '토끼'와 '펭귄' 이미지를 보여주며, 이미지 내에서 해당 객체가 어디에 위치하는지(Bounding Box)와 무엇인지(Class)를 학습시킵니다. 이 프로젝트에서는 Roboflow에 공개된 데이터셋을 사용합니다.
2. **정확도 향상:** 모델은 학습을 반복(Epochs)하면서, 주어진 이미지에서 객체의 특징을 더 잘 찾아내도록 스스로를 개선합니다. 더 많은 데이터와 더 긴 학습 시간은 일반적으로 더 높은 정확도로 이어집니다.
3. **추론 (Inference):** 학습이 완료된 모델은 이제껏 본 적 없는 새로운 이미지나 동영상 프레임이 주어졌을 때, 학습된 지식을 바탕으로 '토끼'나 '펭귄' 객체를 스스로 찾아내고 위치를 표시할 수 있게 됩니다. 이 과정을 '추론'이라고 합니다.

## Roboflow: 데이터셋 준비의 모든 것

**Roboflow**는 컴퓨터 비전 프로젝트를 위한 데이터셋을 쉽게 구축하고 관리할 수 있도록 도와주는 플랫폼입니다. 이미지에 라벨을 붙이는 작업(Annotation)부터, 데이터의 양을 늘리는 증강(Augmentation), 그리고 다양한 포맷으로 데이터셋을 내보내는 기능까지 제공합니다. 이 프로젝트에서는 Roboflow를 통해:

-  공개된 토끼와 펭귄 데이터셋을 다운로드합니다.
-  YOLOv8 학습에 필요한 형식으로 데이터셋을 구성합니다.
-  API를 통해 학습된 모델을 배포하고 추론하는 기능을 활용합니다.

## Example Inference Output

프로젝트를 통해 생성된 객체 탐지 추론 결과의 예시입니다. 로컬 환경과 Roboflow API를 통한 추론 결과를 시각적으로 비교할 수 있습니다.

### 펭귄 탐지 결과

| API 추론                                                                | 로컬 추론                                                                   |
| ----------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| ![Penguin API Inference](@roboflow_yolo/image/penguin_api_detected.png) | ![Penguin Local Inference](@roboflow_yolo/image/penguin_local_detected.png) |

### 토끼 탐지 결과

| API 추론                                                              | 로컬 추론                                                                 |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| ![Rabbit API Inference](@roboflow_yolo/image/rabbit_api_detected.png) | ![Rabbit Local Inference](@roboflow_yolo/image/rabbit_local_detected.png) |

## 성능 비교 지표

**아래 표의 데이터는 예시이며, 실제 프로젝트에서 측정된 정확한 값으로 대체해야 합니다.**

로컬 추론과 Roboflow API 추론 방식의 성능을 정량적으로 비교하여 각 방식의 장단점과 실제 적용 시 고려사항을 분석합니다.

| 모델추론 방식 | mAP50 | FPS | 장점                                    | 단점                                                                |
| ------------- | ----- | --- | --------------------------------------- | ------------------------------------------------------------------- |
| YOLOv8 (로컬) |       |     | 빠르고 무료, 오프라인 가능, 데이터 보안 | 고사양 하드웨어 필요, 복잡한 설치 및 환경 설정                      |
| YOLOv8 (API)  |       |     | 간편한 사용, 일관된 성능 (서버 의존)    | 느림 (네트워크 지연), 비용 발생 가능, 인터넷 필수, 데이터 외부 전송 |

## What I Learned

이 프로젝트를 수행하며 얻은 핵심적인 경험과 기술적 통찰, 그리고 문제 해결 과정을 정리합니다. 이는 단순한 코드 구현을 넘어선 저의 사고력과 학습 능력을 보여줍니다.

-  **Roboflow 활용 능력:** 데이터셋 구축, 라벨링, 증강, 그리고 다양한 형식으로의 내보내기 등 Roboflow 플랫폼을 활용하여 컴퓨터 비전 프로젝트의 데이터 파이프라인을 효율적으로 관리하는 방법을 익혔습니다.
-  **YOLOv8 모델 이해 및 적용:** 최신 YOLOv8 모델의 구조와 학습, 추론 과정을 깊이 이해하고 실제 동영상 데이터에 적용하는 실무 경험을 쌓았습니다.
-  **추론 환경 비교 분석:** 로컬 환경에서의 직접 추론과 클라우드 기반 API 추론의 성능(속도, 정확도) 및 비용, 보안 측면에서의 장단점을 직접 비교 분석하며 각 방식의 적절한 활용 시점을 판단하는 능력을 길렀습니다.
-  **전이 학습의 중요성:** `yolov8n.pt`와 같은 사전 학습 모델을 활용한 전이 학습이 새로운 객체 탐지 모델을 빠르고 효율적으로 구축하는 데 얼마나 중요한지 체감했습니다.
-  **성능 최적화에 대한 고민:** API 추론의 한계(낮은 FPS)를 경험하며, 향후 실시간 성능 개선을 위해 YOLOv8n에서 YOLOv8s와 같은 경량 모델로의 업그레이드, TensorRT와 같은 최적화 기술 적용 등 모델 최적화 방안에 대한 깊은 고민을 시작했습니다.
