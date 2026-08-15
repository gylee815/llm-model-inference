# 단일 LLM 서빙 데모 (Simple LLM Serving Demo)

이 프로젝트는 FastAPI를 사용하여 단일 LLM 모델을 서빙하는 기본 데모입니다. 본 서비스는 `facebook/opt-125m` 모델을 사용하며, 워크로드 관리(Workload Management) 및 별도 프로세스 기반의 모델 실행 구조를 구현합니다. 또한 효율적인 배치 추론을 위해 vLLM 연동을 지원합니다.

## 아키텍처 개요 (Architecture Overview)

시스템은 각 컴포넌트별로 관심사를 분리한 모듈형 아키텍처로 설계되었습니다:

### 핵심 컴포넌트 (Core Components)

#### 1. **main.py (API 레이어)**
- **역할**: HTTP API 엔드포인트 제공 및 요청/응답 처리
- **주요 기능**:
  - REST API 엔드포인트 노출 (`/basic_generate`, `/generate`, `/generate_stream`, `/generate_vllm`)
  - Pydantic 모델을 통한 요청 유효성 검증
  - FastAPI 애플리케이션 라이프사이클 및 의존성 주입 관리
  - 동기식 및 스트리밍 응답 기능 제공

#### 2. **LLMEngine 클래스** (`llm/llm.py`)
- **역할**: 상위 오케스트레이션 및 클라이언트 인터페이스
- **주요 기능**:
  - WorkloadManager와 ModelExecutor 간의 협력 제어
  - 스트리밍 요청을 위한 지속적인 처리 루프 관리
  - 일반 추론 및 vLLM 기반 생성 방식 모두 제공
  - 적절한 큐(Queue) 관리를 통한 비동기 스트리밍 처리
  - 모델 라이프사이클 및 리소스 정리(Cleanup) 관리

#### 3. **WorkloadManager** (`llm/workload_manager.py`)
- **역할**: 요청 큐잉 및 배치 관리
- **주요 기능**:
  - 들어오는 요청 큐 관리 (스트리밍 및 배치용 분리)
  - 처리량(Throughput) 최적화를 위한 배치 로직 구현
  - 활성 시퀀스(Sequence) 상태 추적
  - 생성부터 완료까지 요청 라이프사이클 처리
  - 스트리밍 및 비스트리밍 워크로드 모두 지원

#### 4. **ModelExecutor** (`llm/model_executor.py`)
- **역할**: 프로세스 관리 및 모델 실행 조율
- **주요 기능**:
  - 모델 추론을 위한 별도의 워커 프로세스 관리
  - 큐(Queue)를 통한 프로세스 간 통신(IPC) 처리
  - 메인 프로세스와 모델 워커 간의 동작 조율
  - 배치 및 스트리밍 실행 모드 지원

#### 5. **ModelWorker** (`llm/model_worker.py`)
- **역할**: 별도 프로세스에서 실제 모델 추론 실행
- **주요 기능**:
  - 데이터 및 안정성 격리를 위해 별도 프로세스에서 동작
  - Hugging Face transformers를 이용한 실제 모델 추론 수행
  - 모델 상태 및 토큰 생성 관리
  - 배치 및 스트리밍 토큰 생성 지원
  - 디바이스(CPU/GPU) 관리

#### 6. **ModelManager** (`llm/model_manager.py`)
- **역할**: 모델 로딩 및 캐싱
- **주요 기능**:
  - 트랜스포머 모델 및 토크나이저 로드/캐싱
  - 모델 저장 및 조회 관리
  - 모델 초기화 및 설정 관리

---

## 환경 설정 및 실행 방법 (Setup Options)

환경 및 운영체제 조건에 따라 아래 세 가지 방법 중 하나를 선택하여 진행합니다. (`requirements.txt` 상단 주석 참조)

### [방법 1] 추천 / 최신 방식 (`uv` + Python 3.11)
> `requirements.txt` 파일의 **`## Setup 1 & 2`** 목록을 사용하는 방법입니다.

초고속 파이썬 패키지/버전 관리 도구인 `uv`를 사용하면, 시스템에 파이썬 3.11이 설치되어 있지 않더라도 자동으로 3.11 버전을 가져와 완벽한 격리 환경을 구축합니다.

```bash
# 1. uv를 이용해 파이썬 3.11 가상환경 생성
uv venv .venv --python 3.11

# 2. 가상환경 활성화 (macOS/Linux)
source .venv/bin/activate  # Windows 환경: .venv\Scripts\activate

# 3. uv pip 명령어로 의존성 패키지 초고속 설치
uv pip install -r requirements.txt

# 4. uv를 통해 파이썬 3.11 서버 실행
uv run --python 3.11 python main.py
```

---

### [방법 2] 기본 방식 (`python3` + `venv`)
> `requirements.txt` 파일의 **`## Setup 1 & 2`** 목록을 사용하는 방법입니다.

시스템에 기본으로 설치된 표준 파이썬 3 도구를 사용하는 기존 방식입니다.

```bash
# 1. 시스템 기본 python3 명령어로 가상환경 생성
python3 -m venv venv

# 2. 가상환경 활성화 (macOS/Linux)
source venv/bin/activate  # Windows 환경: venv\Scripts\activate

# 3. pip 명령어로 의존성 패키지 설치
pip install -r requirements.txt

# 4. python3 명령어로 서버 실행
python3 main.py
```

---

### [방법 3] macOS (Apple Silicon) vLLM 소스 직접 빌드 가이드
> `requirements.txt` 파일의 **`## Setup 3`** 주석 목록을 참조하는 방법입니다.

macOS에서는 `pip install -r requirements.txt`만으로 **vLLM이 설치되지 않습니다**.
PyPI에는 Linux+CUDA용 사전빌드 wheel만 올라와 있어서, macOS에서 `pip install vllm`을
실행하면 소스(sdist) 빌드로 넘어가는데, 이 빌드가 `pyproject.toml`에서
`torch==2.6.0`을 고정으로 요구합니다. 그런데 이 torch 버전은 Python 3.13 이상용
macOS wheel이 없어서, 최신 Python을 쓰고 있다면 `No matching distribution found
for torch==2.6.0` 에러로 바로 빌드가 실패합니다.

vLLM은 macOS를 별도의 CPU 백엔드로 지원하지만, 반드시 소스에서 빌드해야 합니다.
아래 절차는 이 저장소 기준으로 macOS 14(Sonoma), Apple Silicon에서 실제로
검증한 내용입니다.

**요구 사항**
- Python 3.9–3.12 (아래는 3.11 기준. torch==2.6.0이 Python 3.13+ 용 macOS wheel을
  제공하지 않아서 vLLM의 macOS CPU 백엔드는 3.13 이상을 지원하지 않습니다)
- Apple Clang 15.x 또는 16.x가 포함된 Xcode Command Line Tools —
  `xcode-select --install` 실행 후 `clang --version`으로 확인
- macOS Sonoma 이상

**설치 절차**
```bash
# 1. 시스템/최신 Python이 아닌 Python 3.11로 venv를 새로 만든다
python3.11 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade   # uv로 만든 venv라면 pip이 없을 수 있어 필요

# 2. 이 프로젝트가 맞춰져 있는 vLLM 버전을 clone하고, macOS용 torch==2.6.0을
#    올바르게 고정해둔 CPU 전용 requirements를 먼저 설치한다.
git clone --depth 1 --branch v0.8.3 https://github.com/vllm-project/vllm.git .build/vllm
pip install -r .build/vllm/requirements/cpu.txt

# 3. requirements/cpu.txt에 빠져 있는 vLLM 자체 빌드 의존성을 설치하고,
#    네이티브 CPU 확장 모듈을 컴파일한다. macOS는 자동 감지되어
#    VLLM_TARGET_DEVICE=cpu가 자동으로 설정된다. 실제 C++ 컴파일이 진행되므로
#    수 분 정도 걸릴 수 있다.
pip install "cmake>=3.26" "setuptools-scm>=8.0" wheel
cd .build/vllm && pip install --no-build-isolation -e . && cd -

# 4. requirements/cpu.txt는 macOS/arm64에서 torchaudio 버전을 고정해두지
#    않아서, torch==2.6.0과 ABI가 맞지 않는 최신 버전이 깔린다
#    (`Symbol not found: _aoti_torch_abi_version` 임포트 에러 발생). 버전을
#    다시 맞춰준다:
pip install "torchaudio==2.6.0"

# 5. 이 프로젝트의 나머지 의존성을 설치한다 (requirements.txt의 Setup 3 참조).
pip install -r requirements.txt
```

`requirements.txt`에는 `transformers<4.52.0`, `httpx<0.28`도 함께 고정해뒀습니다.
vLLM 0.8.3는 transformers 5.x의 토크나이저 API 변경 이전 버전이라 그대로 두면
`AttributeError: GPT2Tokenizer has no attribute all_special_tokens_extended`
에러가 나고, 이 저장소의 테스트 코드는 httpx 0.28에서 제거된
`AsyncClient(app=...)` 인자를 사용합니다.

**알려진 제약사항**: vLLM의 실질적인 성능 이점(paged attention CUDA 커널,
고처리량 연속 배치)은 NVIDIA GPU가 있어야 나옵니다. macOS에서는 CPU 백엔드만
사용할 수 있어서 이 데모의 구조를 학습하는 데는 문제없지만, 속도는 느리고
FP32/FP16만 지원합니다.

---

서비스 실행 후 웹 브라우저에서 **`http://localhost:8080/`**에 접속하여 내장 프론트엔드 UI 플레이그라운드를 이용할 수 있습니다. (포트 변경 시 `PORT=8080 python3 main.py`)

---

## 웹 프론트엔드 플레이그라운드 (Web Frontend Playground)

`http://localhost:8080/` 접속 시 내장된 인터랙티브 웹 UI를 사용할 수 있습니다.

웹 UI 주요 기능:
- **⚡ 스트리밍 모드 (`/generate_stream`)**: 실시간 속도(tokens/sec) 및 TTFT (Time to First Token) 메트릭이 포함된 실시간 토큰 스트리밍.
- **🎯 기본 단일 모드 (`/basic_generate`)**: 응답 지연 시간(Latency)을 확인할 수 있는 단일 프롬프트 추론.
- **📦 배치 생성 모드 (`/generate`)**: 다중 프롬프트 동시 배치 추론 테스트.
- **🚀 vLLM 배치 모드 (`/generate_vllm`)**: vLLM 가속을 활용한 고성능 배치 생성.
- **빠른 프리셋 프롬프트**: 원클릭으로 테스트용 프롬프트 입력 가능.

## API 사용법 (API Usage)

### 기본 단일/배치 생성
`http://localhost:8080/generate`로 POST 요청을 전송합니다:
```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["Hello, I am"]}'
```

### vLLM 배치 생성
vLLM을 활용한 효율적인 배치 추론은 `/generate_vllm` 엔드포인트를 사용합니다:
```bash
curl -X POST http://localhost:8080/generate_vllm \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["Hello, I am", "The weather is", "Once upon a time"]}'
```

### 스트리밍 생성
실시간 토큰 스트리밍 추론:
```bash
curl -X POST http://localhost:8080/generate_stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, I am"}' \
  --no-buffer
```

## 테스트 실행 (Running Tests)

테스트 실행:
```bash
pytest tests/ -v
```

특정 테스트 파일 실행:
```bash
python3 -m pytest tests/test_vllm.py -v
python3 -m pytest tests/test_api.py -v
```

## 주요 특징 (Features)

- **다양한 생성 모드**: 단일 및 배치 처리를 지원하는 기본 텍스트 생성
- **스트리밍 지원**: Server-Sent Events (SSE) 기반 실시간 토큰 스트리밍
- **vLLM 연동**: vLLM을 활용한 고성능 배치 추론
- **프로세스 격리**: 모델 실행을 별도 프로세스에서 수행하여 시스템 안정성 확보
- **워크로드 관리**: 지능형 배치 처리 및 큐 관리
- **종합 테스트 구축**: 주요 엔드포인트 및 기능에 대한 테스트 모듈 포함

## 아키텍처 장점 (Architecture Benefits)

- **확장성 (Scalability)**: 별도 프로세스 분리로 리소스 활용 극대화
- **신뢰성 (Reliability)**: 프로세스 격리로 모델 오류 시 API 메인 서비스 영향 최소화
- **성능 (Performance)**: 배치 및 vLLM 연동을 통한 처리량 최적화
- **유연성 (Flexibility)**: 스트리밍 및 배치 처리 모드 완벽 지원
