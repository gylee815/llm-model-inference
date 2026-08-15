# 다중 모델 서빙 데모 (Multi-Model Serving Demo)

이 프로젝트는 제약된 리소스 환경에서 여러 ML/DL 모델을 효율적으로 관리 및 서빙하는 다중 모델 서빙(Multi-Model Serving) 데모입니다. 최대 2개의 모델을 메모리에 유지하는 모델 캐시를 구현하여, 요청 패턴에 따라 온디맨드(On-Demand) 방식으로 모델을 로드 및 언로드(LRU 캐시 알고리즘 기반)합니다.

## 주요 기능 (Features)

- **온디맨드 모델 로딩 (On-demand loading)**: 요청 발생 시 필요한 모델을 메모리에 로드
- **LRU (Least Recently Used) 모델 캐싱**: 메모리 한도(기본 2개) 초과 시 가장 오래 사용되지 않은 모델 자동 언로드
- **대화형 웹 UI 플레이그라운드**: 내장 웹 인터페이스(`http://localhost:8001/`)를 통해 실시간 LRU 캐시 상태 모니터링 및 추론 실습
- **다양한 모델 타입 지원**: 텍스트(감정 분석, 스팸 감지) 및 이미지(이미지 분류) 모델 지원
- **범용 API 인터페이스**: 모델 입력 형태에 관계없이 통일된 요청 엔드포인트 제공
- **프레임워크별 워커 구현**: Transformers, TorchVision, Triton 등 개별 워커 모듈 분리

## 프로젝트 구조 (Project Structure)

```
.
├── app/
│   ├── __init__.py
│   ├── server.py      # FastAPI 서버, REST API 및 프론트엔드 엔드포인트
│   ├── store.py       # 모델 메타데이터 및 설정 관리
│   ├── manager.py     # LRU 모델 캐싱 및 라이프사이클 관리
│   ├── engine.py      # 모델 워커 팩토리 및 워커 생성
│   └── worker.py      # 추상 워커 클래스 및 프레임워크별 추론 구현
├── config/
│   └── models.json    # 모델 메타데이터 설정 파일
├── static/
│   └── index.html     # 대화형 인터랙티브 웹 UI 플레이그라운드
└── requirements.txt   # 프로젝트 의존성 패키지
```

## 실행 방법 (Quick Start Guide)

### 1. 가상환경 구축 및 패키지 설치

`python3`를 사용하여 가상환경을 생성 및 활성화한 뒤 의존성 패키지를 설치합니다:

```bash
# multi_model_serving 디렉토리로 이동
cd ch03/multi_model_serving

# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화 (macOS/Linux)
source venv/bin/activate

# 가상환경 활성화 (Windows)
# venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 백엔드 서버 및 웹 UI 실행

`python3` 모듈 실행 명령어로 백엔드 서버를 시작합니다:

```bash
# 기본 포트(8001)로 실행
python3 -m app.server

# 또는 특정 포트(예: 8081)로 지정하여 실행
PORT=8081 python3 -m app.server
```

서버가 실행되면 웹 브라우저를 열고 다음 주소로 접속합니다:
👉 **`http://localhost:8001/`** (또는 포트를 지정한 경우 `http://localhost:8081/`)

### 3. 웹 프론트엔드 플레이그라운드 주요 기능

- **LRU 캐시 모니터링**: 현재 메모리에 로드된 모델(최대 2개 한도) 상태를 실시간 배지로 확인
- **모델 선택 및 메타데이터 확인**: 감정 분석, 스팸 감지, 이미지 분류 모델 선택
- **원클릭 프리셋 입력**: 테스트용 샘플 데이터 원클릭 입력
- **추론 실행 및 실시간 캐시 갱신**: 추론 요청 시 지연 시간(ms) 측정 및 모델 온디맨드 로딩에 따른 LRU 캐시 상태 변화 실시간 시각화

---

## Triton Server 설정 (선택 사항)

Triton Inference Server 연동 테스트 방법:

1. 모델 리포지토리 디렉토리 구조 생성:
```bash
mkdir -p model_dir/densenet_onnx/1
```

2. ONNX 모델 파일 복사:
```bash
cp path/to/your/model.onnx model_dir/densenet_onnx/1/
```

3. 모델 설정 파일 `model_dir/densenet_onnx/config.pbtxt` 작성:
```protobuf
name: "densenet_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 0
input [
  {
    name: "data_0"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]
output [
  {
    name: "fc6_1"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]
```

4. Triton Docker 컨테이너 실행:
```bash
docker run -p8009:8000 -p8010:8001 -p8011:8002 \
    -v $(pwd)/model_dir:/models \
    nvcr.io/nvidia/tritonserver:24.12-py3 \
    tritonserver --model-repository=/models --model-control-mode=explicit
```

5. 테스트 실행:
```bash
python3 -m unittest tests/test_triton_densenet.py
```

---

## API 사용법 (CLI)

### 등록된 모델 및 캐시 상태 조회
```bash
curl http://localhost:8001/models
```

### 모델 추론 요청 (Predictions)

텍스트 감정 분석 모델 (Sentiment Analysis):
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "550e8400-e29b-41d4-a716-446655440000", "input_data": "This movie was great!"}'
```

스팸 감지 모델 (Spam Detection):
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8", "input_data": "Win a free iPhone now!"}'
```

이미지 분류 모델 (Image Classification):
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7", "input_data": "tests/images/cat1.jpg"}'
```

---

## 시스템 아키텍처 (Architecture)

1. **Server** (`server.py`): HTTP REST API 엔드포인트 및 정적 파일(웹 UI) 제공
2. **Store** (`store.py`): `models.json` 기반 메타데이터 관리
3. **Manager** (`manager.py`): LRU 기반 모델 캐싱 및 메모리 라이프사이클 관리
4. **Engine** (`engine.py`): 프레임워크 타입에 따른 모델 워커 생성 (Factory 패턴)
5. **Worker** (`worker.py`): 추상 워커 인터페이스 및 프레임워크별 추론 실행 (Strategy 패턴)
   - `TransformerWorker`: Hugging Face Transformers 기반 모델 처리
   - `TorchVisionWorker`: PyTorch TorchVision 기반 이미지 모델 처리
   - `TritonWorker`: Triton Inference Server 기반 추론 처리
