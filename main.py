import secrets
import torch
from typing import Annotated, Literal
from fastapi import FastAPI, Form, UploadFile, File, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image

# Load the Hugging Face model and feature extractor
model_name = "codewithdark/vit-chest-xray"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

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
    predicted_class = logits.argmax(-1).item()
    confidence = torch.nn.functional.softmax(logits, dim=-1)[0, predicted_class].item()

    # Hugging Face model label mapping
    labels = model.config.id2label
    # prediction_label = labels[predicted_class]
    label_mapping = {
        0: "Cardiomegaly",
        1: "Edema",
        2: "Consolidation",
        3: "Pneumonia",
        4: "No Finding",
    }

    prediction_label = label_mapping.get(predicted_class, f"Unknown Label ({predicted_class})")


    
    base_url = request.base_url
    return {
        "prediction": prediction_label,
        "confidence": round(confidence, 4),
        "processed_image_link": f"{base_url}download_processed_image/{image_id}",
    }

@app.get("/download_processed_image/{image_id}", summary="Download Processed X-ray Image")
async def download_processed_image(image_id: str):
    """Serve the processed X-ray image for download."""
    return FileResponse(f"data/{image_id}_processed.png", media_type="image/png", filename="processed_xray.png")