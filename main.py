import io
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

app = FastAPI()
model = tf.keras.models.load_model('model/')
class_names = ["Common Rust", "Gray Leaf Spot", "Healthy", "Northern Leaf Blight"]
new_size = (256, 256)


@app.get("/api/ping", response_class=HTMLResponse)
async def ping():
    return "It works!"


@app.post("/api/predict", response_class=JSONResponse)
async def predict(file: UploadFile = File(...)):
    f = await file.read()
    img = Image.open(io.BytesIO(f)).resize(new_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    return {
        "class": predicted_class,
        "confidence": confidence
    }