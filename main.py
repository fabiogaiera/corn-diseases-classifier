import io
import numpy as np
import tflite_runtime.interpreter as tflite

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# TensorFlow Lite initialization

interpreter = tflite.Interpreter(model_path='static/model/model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ["Common Rust", "Gray Leaf Spot", "Healthy", "Northern Leaf Blight"]

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

new_size = (256, 256)


@app.get("/api/ping", response_class=HTMLResponse)
async def ping():
    return "It works!"


@app.post("/api/predict", response_class=JSONResponse)
async def predict(file: UploadFile = File(...)):
    
    f = await file.read()
    img = Image.open(io.BytesIO(f)).resize(new_size)

    input_shape = input_details[0]['shape']
    input_data = np.array(img).reshape(input_shape).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index']).reshape(4, )    

    return {
        "class": class_names[np.argmax(output_data)]
    }
