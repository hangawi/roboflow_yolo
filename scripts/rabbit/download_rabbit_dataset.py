from roboflow import Roboflow

# Roboflow API 키를 여기에 입력하세요.
# 이 키는 민감한 정보이므로, 실제 프로젝트에서는 환경 변수 등으로 관리하는 것이 좋습니다.
rf = Roboflow(api_key="UgF3Ulak7a0s9TsaO1jE")

# 프로젝트 및 버전 정보
# 이 정보는 Roboflow 웹사이트에서 확인할 수 있습니다.
project = rf.workspace("lobo-v5dzy").project("rabbit-qmql6-speqw")
version = project.version(1)

# 데이터셋 다운로드
# 'yolov8' 형식으로 다운로드합니다.
print("데이터셋 다운로드를 시작합니다...")
dataset = version.download("yolov8", location=r"../../data/rabbit_dataset")
print("데이터셋 다운로드가 완료되었습니다.")
print(f"데이터셋이 다음 경로에 저장되었습니다: {dataset.location}")
