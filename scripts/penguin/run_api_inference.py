import cv2
from roboflow import Roboflow
import os
import time

# --- 1. 설정 ---
ROBOFLOW_API_KEY = "UgF3Ulak7a0s9TsaO1jE"
INPUT_VIDEO_PATH = "../../raw_videos/penguin_original.mp4"
OUTPUT_VIDEO_PATH = "../../results/penguin_api_detected.mp4"
TEMP_FRAME_PATH = "temp_frame.jpg" # API에 보낼 임시 이미지 파일

# --- 2. Roboflow 모델 초기화 ---
try:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace("lobo-v5dzy").project("penguindataset-4dujc-quofz")
    model = project.version(1).model
except Exception as e:
    print(f"Roboflow 모델 초기화 중 오류 발생: {e}")
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

print("\nRoboflow API를 사용하여 동영상 처리를 시작합니다.")
print("주의: 이 작업은 매우 느릴 수 있으며, Roboflow API 사용량 제한에 영향을 줄 수 있습니다.")

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임을 임시 이미지 파일로 저장
    cv2.imwrite(TEMP_FRAME_PATH, frame)

    try:
        # Roboflow API로 예측 수행 (신뢰도 70% 이상만)
        prediction = model.predict(TEMP_FRAME_PATH, confidence=70, overlap=30).json()
        
        # 예측 결과를 프레임에 그리기
        for box in prediction['predictions']:
            x0 = int(box['x'] - box['width'] / 2)
            x1 = int(box['x'] + box['width'] / 2)
            y0 = int(box['y'] - box['height'] / 2)
            y1 = int(box['y'] + box['height'] / 2)
            
            class_name = box['class']
            confidence = box['confidence']
            
            # 네모 박스 그리기 (녹색)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            
            # 라벨 텍스트 생성
            label = f"{class_name}: {confidence:.2f}"
            
            # 라벨 배경용 사각형 그리기
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x0, y0 - 20), (x0 + w, y0), (0, 255, 0), -1)
            
            # 라벨 텍스트 쓰기 (검은색)
            cv2.putText(frame, label, (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    except Exception as e:
        print(f"프레임 {frame_count} 처리 중 오류 발생: {e}")
        time.sleep(1)

    # 처리된 프레임을 결과 동영상에 쓰기
    out.write(frame)
    
    frame_count += 1
    print(f"  {frame_count} / {total_frames} 프레임 처리 완료")

# --- 4. 종료 및 정리 ---
print("\n동영상 처리가 완료되었습니다.")
print(f"결과가 다음 파일로 저장되었습니다: {OUTPUT_VIDEO_PATH}")

cap.release()
out.release()
cv2.destroyAllWindows()

# 임시 파일 삭제
if os.path.exists(TEMP_FRAME_PATH):
    os.remove(TEMP_FRAME_PATH)
