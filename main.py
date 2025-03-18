import os
import secrets
import torch
from typing import Annotated, Literal
from fastapi import FastAPI, Form, UploadFile, File, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image

# Define model storage directory
MODEL_DIR = "models/vit-xray"
REQUIRED_FILES = ["config.json", "pytorch_model.bin", "preprocessor_config.json"]

# Check if model files exist; if not, download them
missing_files = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(MODEL_DIR, f))]
if missing_files:
    print("Some model files are missing. Downloading...")
    
    model_name = "codewithdark/vit-chest-xray"
    model = ViTForImageClassification.from_pretrained(model_name)
    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

    model.save_pretrained(MODEL_DIR)
    feature_extractor.save_pretrained(MODEL_DIR)
    
    print("Model downloaded and saved!")

# Load the model from the local directory
feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_DIR)
model = ViTForImageClassification.from_pretrained(MODEL_DIR)

# Define the label mapping
label_mapping = {
    0: "Cardiomegaly",
    1: "Edema",
    2: "Consolidation",
    3: "Pneumonia",
    4: "No Finding",
}

app = FastAPI(
    title="ViT-Xray",
    description="AI-powered chest X-ray analysis using a Vision Transformer (ViT) model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post(
    "/analyze_xray/",
    summary="Analyze Chest X-ray",
    description="Accepts a chest X-ray image, runs it through a ViT model, and returns a classification result.",
)
def analyze_xray(
    request: Request,
    image: Annotated[UploadFile, File()],
    analysis_type: Annotated[
        Literal["Standard", "Detailed"],
        Form(...)
    ] = "Standard",  # Default to "Standard" analysis
):
    """Processes an X-ray image and predicts abnormalities using ViT."""

    # Generate a unique ID for the image
    image_id = secrets.token_hex(16)
    input_file = f"data/{image_id}_input.png"
    output_file = f"data/{image_id}_processed.png"

    # Save the uploaded image
    with open(input_file, "wb") as f:
        f.write(image.file.read())

    # Open and preprocess image
    img = Image.open(input_file).convert("RGB")
    img = img.resize((224, 224))  # Resize for ViT input
    img.save(output_file)  # Save processed image
    
    # Convert image to tensor for model inference
    inputs = feature_extractor(images=img, return_tensors="pt")

    # Run model inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted class and confidence
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    confidence = torch.nn.functional.softmax(logits, dim=-1)[0, predicted_class_idx].item()

    # Get the corresponding label
    predicted_label = label_mapping.get(predicted_class_idx, f"Unknown Label ({predicted_class_idx})")

    # Prepare response
    base_url = request.base_url if request else "http://localhost:8000"
    response = {
        "prediction": predicted_label,
        "confidence": round(confidence, 4),
        "processed_image_link": f"{base_url}download_processed_image/{image_id}",
    }

    # If detailed analysis is requested, return all class probabilities
    if analysis_type == "Detailed":
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0].tolist()
        detailed_scores = {label_mapping[i]: round(probabilities[i], 4) for i in range(len(label_mapping))}
        response["detailed_scores"] = detailed_scores

    return response

@app.get("/download_processed_image/{image_id}", summary="Download Processed X-ray Image")
async def download_processed_image(image_id: str):
    """Serve the processed X-ray image for download."""
    return FileResponse(f"data/{image_id}_processed.png", media_type="image/png", filename="processed_xray.png")
