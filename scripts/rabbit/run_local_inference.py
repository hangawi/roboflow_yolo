import cv2
from ultralytics import YOLO
import os

# --- 1. 설정 ---

# 사용할 모델 가중치 파일의 전체 경로
MODEL_WEIGHTS_PATH = "../../trained_models/rabbit_trained_model.pt"

# 처리할 동영상 파일의 전체 경로
INPUT_VIDEO_PATH = "../../raw_videos/rabbit_original.mp4"

# 결과 동영상을 저장할 전체 경로
OUTPUT_VIDEO_PATH = "../../results/rabbit_local_detected.mp4"

# --- 설정 끝 ---

# --- 2. YOLO 모델 로드 ---
try:
    model = YOLO(MODEL_WEIGHTS_PATH)
except Exception as e:
    print(f"YOLO 모델 로드 중 오류 발생: {e}")
    exit()

# --- 3. 동영상 처리 ---
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    print(f"오류: 동영상 파일을 열 수 없습니다. 경로를 확인하세요: {INPUT_VIDEO_PATH}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

print("\n로컬 모델을 사용하여 동영상 처리를 시작합니다...")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 현재 프레임에서 객체 탐지 수행
    results = model(frame, verbose=False) 

    # 탐지 결과가 하나 이상인 경우 results[0] 사용
    if results:
        annotated_frame = results[0].plot() # plot() 함수가 바운딩 박스를 그려줌
    else:
        annotated_frame = frame

    # 처리된 프레임을 결과 동영상에 쓰기
    out.write(annotated_frame)
    
    frame_count += 1
    # 100 프레임마다 진행 상황 출력
    if frame_count % 100 == 0:
        print(f"  {frame_count} / {total_frames} 프레임 처리 완료")


# --- 4. 종료 및 정리 ---
print("\n동영상 처리가 완료되었습니다.")
print(f"결과가 다음 파일로 저장되었습니다: {OUTPUT_VIDEO_PATH}")

cap.release()
out.release()
cv2.destroyAllWindows()
