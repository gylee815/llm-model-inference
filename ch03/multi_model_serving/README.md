# Multi-Model Serving Demo

This is a simple demonstration of a multi-model serving service that manages multiple ML models with limited resources. The service implements a model cache that can hold up to 2 models at a time, loading and unloading models on-demand based on usage patterns (LRU cache).

## Features

- On-demand model loading
- LRU (Least Recently Used) model caching
- Interactive Web UI Playground (`http://localhost:8001/`)
- Support for different model types (text and image)
- Generic API interface for different model inputs
- Model metadata management
- Framework-specific model workers (Transformers, TorchVision, Triton)

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── server.py      # FastAPI server, REST API, and static frontend endpoints
│   ├── store.py       # Model metadata management
│   ├── manager.py     # Model caching and lifecycle (LRU)
│   ├── engine.py      # Model worker factory and management
│   └── worker.py      # Abstract worker and framework-specific implementations
├── config/
│   └── models.json    # Model configurations
├── static/
│   └── index.html     # Interactive Web UI Playground
└── requirements.txt   # Project dependencies
```

## How to Run (Quick Start Guide)

### 1. Environment Setup

Create and activate a virtual environment using `python3`, then install dependencies:

```bash
# Navigate to multi_model_serving directory
cd ch03/multi_model_serving

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Backend Server & Web UI

Start the service using `python3`:

```bash
# Run on default port (8001)
python3 -m app.server

# Or specify a custom port (e.g. 8081)
PORT=8081 python3 -m app.server
```

Once started, open your browser and navigate to:
👉 **`http://localhost:8001/`** (or `http://localhost:8081/` if using custom port)

### 3. Web UI Playground Features

- **LRU Model Cache Monitor**: Visual live status of currently loaded models in memory (capacity limit: 2).
- **Model Selector**: Easily select between sentiment analysis, spam detection, or image classification models.
- **Quick Preset Inputs**: One-click prompt fill for instant testing.
- **Live Latency & Output**: Measures inference execution time (ms) and displays raw JSON response while updating the cache status.

---

## Triton Server Setup (Optional)

To run the tests with Triton Inference Server:

1. Create a model repository directory structure:
```bash
mkdir -p model_dir/densenet_onnx/1
```

2. Copy your ONNX model to the repository:
```bash
cp path/to/your/model.onnx model_dir/densenet_onnx/1/
```

3. Create a model configuration file `model_dir/densenet_onnx/config.pbtxt`:
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

4. Start Triton server with explicit model control:
```bash
# Using Docker (recommended)
docker run -p8009:8000 -p8010:8001 -p8011:8002 \
    -v $(pwd)/model_dir:/models \
    nvcr.io/nvidia/tritonserver:24.12-py3 \
    tritonserver --model-repository=/models --model-control-mode=explicit
```

5. Run the tests:
```bash
python3 -m unittest tests/test_triton_densenet.py
```

---

## API Usage

### List Available Models & Cache Status
```bash
curl http://localhost:8001/models
```

### Make Predictions via CLI

For text sentiment analysis:
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "550e8400-e29b-41d4-a716-446655440000", "input_data": "This movie was great!"}'
```

For spam detection:
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8", "input_data": "Win a free iPhone now!"}'
```

For image classification:
```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"model_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7", "input_data": "tests/images/cat1.jpg"}'
```

---

## Architecture

The service consists of five main components:

1. **Server** (`server.py`): Provides HTTP endpoints and static Web UI for model predictions.
2. **Store** (`store.py`): Manages model metadata and configurations from `models.json`.
3. **Manager** (`manager.py`): Handles model caching and LRU lifecycle.
4. **Engine** (`engine.py`): Factory for creating and managing model workers based on framework type.
5. **Worker** (`worker.py`): Abstract base class and framework-specific implementations for model inference.
   - `ModelWorker`: Abstract base class defining the interface.
   - `TransformerWorker`: Handles transformer-based models (Hugging Face).
   - `TorchVisionWorker`: Handles torchvision-based models.
   - `TritonWorker`: Handles Triton Inference Server models.
