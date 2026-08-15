from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Dict, Optional
import uvicorn
import os
from .store import ModelStore
from .manager import ModelManager

app = FastAPI(title="Multi-Model Serving Demo")

# Enable CORS for frontend UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files setup
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def serve_frontend():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Multi-Model Serving Backend is running. Frontend UI not found."}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# Initialize components
model_store = ModelStore("config/models.json")
model_manager = ModelManager(model_store)

class PredictionRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_id: str
    input_data: Any

@app.post("/predict")
async def predict(request: PredictionRequest):
    # Get model worker
    worker = model_manager.get_model_worker(request.model_id)
    if not worker:
        raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
    
    # Make prediction
    try:
        result = worker.predict(request.input_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    return {
        "available_models": model_store.list_models(),
        "loaded_models": model_manager.list_loaded_models(),
        "max_capacity": model_manager.max_models
    }

class CapacityRequest(BaseModel):
    max_capacity: int = Field(ge=2, le=5)

@app.post("/cache/clear")
async def clear_cache():
    model_manager.clear_cache()
    return {
        "status": "cleared",
        "loaded_models": model_manager.list_loaded_models(),
        "max_capacity": model_manager.max_models
    }

@app.post("/cache/capacity")
async def set_cache_capacity(request: CapacityRequest):
    model_manager.set_max_models(request.max_capacity)
    return {
        "status": "ok",
        "max_capacity": model_manager.max_models,
        "loaded_models": model_manager.list_loaded_models()
    }

if __name__ == "__main__":
    # Get port from environment variable or use default 8001
    port = int(os.getenv("PORT", "8001"))
    print(f"Starting Multi-Model Serving backend on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port) 