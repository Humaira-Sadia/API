from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# Load the model
model = tf.keras.models.load_model('C:/Users/Humaira Sadia/Desktop/Cancer_BUSI/model.h5')
print("Model loaded successfully")

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to restrict origins if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)


@app.get("/")
def read_root():
    return {"message": "Breast Cancer Detection"}

# Helper function to preprocess the image
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        processed_image = preprocess_image(image)
        
        predictions = model.predict(processed_image)

        # Assuming the model has 3 output classes
        class_names = ["Benign", "Malignant", "Normal"]
        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions))
        
        # Return the result as JSON
        return JSONResponse({
            "predicted_class": predicted_class,
            "confidence": confidence,
            # "raw_predictions": predictions.tolist()[0]
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
