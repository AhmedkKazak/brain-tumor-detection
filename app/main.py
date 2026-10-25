import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import io
import time
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from tensorflow import keras

from app.config import (
    FRAMEWORK,
    MODEL_PATH,
    CLASS_NAMES,
    IMAGE_SIZE
)


# =====================================
# 1. Project Configuration
# =====================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =====================================
# 2. Load Model Once
# =====================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    model_file = PROJECT_ROOT / MODEL_PATH

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model not found: {model_file}"
        )

    app.state.model = keras.models.load_model(
        str(model_file),
        compile=False
    )

    print("TensorFlow model loaded successfully!")

    yield

    app.state.model = None


# =====================================
# 3. Initialize FastAPI
# =====================================

app = FastAPI(
    title="Brain Tumor Detection API - T35",
    description="MRI Brain Tumor Classification using TensorFlow ResNet50",
    version="1.0.0",
    lifespan=lifespan
)


# =====================================
# 4. Health Endpoint
# =====================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "framework": FRAMEWORK,
        "classes": CLASS_NAMES
    }


# =====================================
# 5. Prediction Endpoint
# =====================================

@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...)
):

    start_time = time.perf_counter()

    contents = await file.read()
    await file.close()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Empty image file"
        )

    # Validate image
    try:
        with Image.open(io.BytesIO(contents)) as img:

            if img.format not in ("JPEG", "PNG"):
                raise HTTPException(
                    status_code=400,
                    detail="Only JPEG and PNG images are supported"
                )

            image = np.asarray(
                img.convert("RGB"),
                dtype=np.uint8
            )

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file"
        )

    # Resize - same preprocessing as training
    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        method="bilinear"
    )

    # ResNet50 preprocessing
    image = keras.applications.resnet50.preprocess_input(
        image.numpy()
    )

    image = np.expand_dims(image, axis=0)

    # Model Prediction
    with tf.device("/CPU:0"):
        predictions = request.app.state.model(
            image,
            training=False
        ).numpy()[0]

    predicted_index = int(np.argmax(predictions))

    # Calculate Latency
    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    # Response
    return {
        "framework": FRAMEWORK,
        "label": CLASS_NAMES[predicted_index],
        "confidence": float(predictions[predicted_index]),
        "probabilities": {
            name: float(predictions[i])
            for i, name in enumerate(CLASS_NAMES)
        },
        "latency_ms": round(latency_ms, 2)
    }