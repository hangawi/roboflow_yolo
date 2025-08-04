from ultralytics import YOLO

# YOLOv8 모델 로드 (사전 학습된 모델)
model = YOLO('yolov8n.pt') 

# 모델 학습 (토끼 데이터셋으로 학습)
# data.yaml 파일의 경로를 정확하게 지정해야 합니다.
# epochs는 학습 횟수를 의미하며, 필요에 따라 조절할 수 있습니다.
results = model.train(data='../data/rabbit_dataset/data.yaml', epochs=50, imgsz=640, project='../trained_models', name='rabbit_trained_model')

print("\n모델 학습이 완료되었습니다.")
print("'runs/detect/train/weights/best.pt' 경로에서 학습된 모델을 확인하세요.")
