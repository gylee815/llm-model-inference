# Simple LLM Serving Demo

This is a simple demonstration of how to serve a single LLM model using FastAPI. The service uses the facebook/opt-125m model and implements a basic serving architecture with workload management and model execution in a separate process. It also includes vLLM integration for efficient batched inference.

## Architecture Overview

The system is designed with a modular architecture that separates concerns across different components:

### Core Components

#### 1. **main.py (API Layer)**
- **Responsibility**: HTTP API endpoints and request/response handling
- **Key Functions**:
  - Exposes REST API endpoints (`/basic_generate`, `/generate`, `/generate_stream`, `/generate_vllm`)
  - Handles request validation using Pydantic models
  - Manages FastAPI application lifecycle and dependency injection
  - Provides both synchronous and streaming response capabilities

#### 2. **LLMEngine Class** (`llm/llm.py`)
- **Responsibility**: High-level orchestration and client interface
- **Key Functions**:
  - Coordinates between WorkloadManager and ModelExecutor
  - Manages the continuous processing loop for streaming requests
  - Provides both traditional and vLLM-based generation methods
  - Handles async streaming with proper queue management
  - Manages model lifecycle and cleanup

#### 3. **WorkloadManager** (`llm/workload_manager.py`)
- **Responsibility**: Request queuing and batch management
- **Key Functions**:
  - Manages incoming request queues (separate for streaming and batch)
  - Implements batching logic to optimize throughput
  - Tracks active sequences and their states
  - Handles request lifecycle from creation to completion
  - Supports both streaming and non-streaming workloads

#### 4. **ModelExecutor** (`llm/model_executor.py`)
- **Responsibility**: Process management and model execution coordination
- **Key Functions**:
  - Manages separate worker processes for model inference
  - Handles inter-process communication via queues
  - Coordinates between main process and model worker
  - Supports both batch and streaming execution modes

#### 5. **ModelWorker** (`llm/model_worker.py`)
- **Responsibility**: Model inference execution in separate process
- **Key Functions**:
  - Runs in a separate process for isolation
  - Handles actual model inference using transformers
  - Manages model state and token generation
  - Supports both batch and streaming token generation
  - Handles device management (CPU/GPU)

#### 6. **ModelManager** (`llm/model_manager.py`)
- **Responsibility**: Model loading and caching
- **Key Functions**:
  - Loads and caches transformer models and tokenizers
  - Manages model storage and retrieval
  - Handles model initialization and configuration

## Setup

1. Create a virtual environment using `python3`:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### macOS (Apple Silicon): vLLM must be built from source

`pip install -r requirements.txt` does **not** install vLLM on macOS. PyPI only
ships pre-built vLLM wheels for Linux+CUDA, so on macOS `pip install vllm` falls
back to building vLLM's sdist — and that build hard-pins `torch==2.6.0` in its
`pyproject.toml`. That torch build has no macOS wheel for Python 3.13+, so the
build fails immediately with `No matching distribution found for torch==2.6.0`
if you're on a newer Python.

vLLM does support macOS via a separate CPU backend, but it must be compiled
from source. These steps were verified against this repo on macOS 14 (Sonoma),
Apple Silicon:

**Requirements**
- Python 3.9–3.12 (3.11 used below; vLLM's macOS CPU backend does not support 3.13+, since torch==2.6.0 has no macOS wheel for it)
- Xcode Command Line Tools with Apple Clang 15.x or 16.x — `xcode-select --install`, then check with `clang --version`
- macOS Sonoma or later

**Steps**
```bash
# 1. Recreate the venv with Python 3.11 (not your system/latest Python)
python3.11 -m venv .venv
source .venv/bin/activate
python -m ensurepip --upgrade   # needed if using a `uv`-created venv

# 2. Clone the vLLM version this project targets and install its CPU-specific
#    requirements first — these correctly pin torch==2.6.0 for macOS.
git clone --depth 1 --branch v0.8.3 https://github.com/vllm-project/vllm.git .build/vllm
pip install -r .build/vllm/requirements/cpu.txt

# 3. Install vLLM's own build dependencies (not covered by requirements/cpu.txt)
#    and compile the native CPU extension. macOS is auto-detected, so
#    VLLM_TARGET_DEVICE=cpu is set automatically. This runs a real C++ build
#    and can take several minutes.
pip install "cmake>=3.26" "setuptools-scm>=8.0" wheel
cd .build/vllm && pip install --no-build-isolation -e . && cd -

# 4. requirements/cpu.txt leaves torchaudio unpinned on macOS/arm64, which
#    resolves to a version that's ABI-incompatible with torch==2.6.0
#    (`Symbol not found: _aoti_torch_abi_version` on import). Pin it back down:
pip install "torchaudio==2.6.0"

# 5. Install the rest of this project's requirements (the vLLM line in
#    requirements.txt is commented out on purpose — it's already installed
#    from source above).
pip install -r requirements.txt
```

`requirements.txt` also pins `transformers<4.52.0` and `httpx<0.28`: vLLM 0.8.3
predates the tokenizer API changes in transformers 5.x (fails with
`AttributeError: GPT2Tokenizer has no attribute all_special_tokens_extended`
otherwise), and this repo's tests use the `AsyncClient(app=...)` argument that
httpx removed in 0.28.

**Known limitation**: real vLLM performance benefits (paged attention CUDA
kernels, high-throughput continuous batching) require an NVIDIA GPU. On macOS
you only get the CPU backend, which works for learning the architecture in
this demo but is slow and limited to FP32/FP16.

## Running the Service

Start the service using `python3`:
```bash
python3 main.py

# Or specify a custom port
PORT=8080 python3 main.py
```

The service will be available at http://localhost:8080 (or your configured `PORT`).

**macOS note**: vLLM's CPU backend starts a `torch.distributed` process group even
for a single worker, and on a machine connected to a corporate VPN/network it can
try to bind to that network's IP and hang indefinitely, retrying every ~75s with
`[c10d] The server socket on [...]:PORT has timed out, will retry.` in the logs.
If startup hangs there, force it onto loopback:
```bash
VLLM_HOST_IP=127.0.0.1 python3 main.py
```

## Web Frontend Playground

A built-in interactive web UI is available at `http://localhost:8080/`.

Features in the Web UI:
- **⚡ Streaming Mode (`/generate_stream`)**: Real-time token streaming with live speed (tokens/sec) and TTFT (Time to First Token) metrics.
- **🎯 Basic Single Mode (`/basic_generate`)**: Single prompt execution with response latency.
- **📦 Batch Generation (`/generate`)**: Multi-prompt batch inference test.
- **🚀 vLLM Batch (`/generate_vllm`)**: vLLM accelerated batch generation.
- **Quick Preset Prompts**: One-click prompt fill for fast testing.

## API Usage

### Basic Generation
Send a POST request to `/generate` with a JSON body:
```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["Hello, I am"]}'
```

### vLLM Generation
For efficient batched inference using vLLM, use the `/generate_vllm` endpoint:
```bash
curl -X POST http://localhost:8080/generate_vllm \
  -H "Content-Type: application/json" \
  -d '{"prompts": ["Hello, I am", "The weather is", "Once upon a time"]}'
```

### Streaming Generation
For real-time token streaming:
```bash
curl -X POST http://localhost:8080/generate_stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, I am"}' \
  --no-buffer
```

## Running Tests

Run the tests with:
```bash
pytest tests/ -v
```

Or run specific test files:
```bash
python3 -m pytest tests/test_vllm.py -v
python3 -m pytest tests/test_api.py -v
```

## Features

- **Multi-modal Generation**: Basic text generation with single and batch processing
- **Streaming Support**: Real-time token streaming with Server-Sent Events
- **vLLM Integration**: High-performance batched inference using vLLM
- **Process Isolation**: Model execution in separate processes for stability
- **Workload Management**: Intelligent batching and queue management
- **Comprehensive Testing**: Full test coverage for all endpoints and functionality

## Architecture Benefits

- **Scalability**: Separate processes allow for better resource utilization
- **Reliability**: Process isolation prevents model crashes from affecting the API
- **Performance**: Batching and vLLM integration optimize throughput
- **Flexibility**: Support for both streaming and batch processing modes 
