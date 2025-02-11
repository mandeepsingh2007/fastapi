from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import uvicorn
from io import BytesIO
from PIL import Image
import pickle
import requests
import threading

app = FastAPI()

# Load the trained model
model = pickle.load(open('pneumonia_mode.sav', 'rb'))

IMAGE_SIZE = (150, 150)

def preprocess_image(image_data):
    img = Image.open(BytesIO(image_data)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    img_array = preprocess_image(image_data)
    probability = model.predict(img_array)[0][0]

    label = "Pneumonia" if probability > 0.5 else "Normal"
    confidence = float(probability) if probability > 0.5 else float(1 - probability)

    return {"prediction": label, "confidence": confidence}

# Start FastAPI server in a separate thread to prevent event loop issues
def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # Run the server in a separate thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # Wait a moment to ensure the server starts
    import time
    time.sleep(2)

    # Send a request to the API
    url = "http://127.0.0.1:8000/predict"
    file_path = "/Users/singh/Downloads/images.jpg"
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)

    print(response.json())
