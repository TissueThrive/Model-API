import os
from fastapi import APIRouter, UploadFile, File, Response
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

router = APIRouter()

model_path = os.path.join(os.path.dirname(__file__), '../models/callus_identify_model.pt')
model = YOLO(model_path)

# Adjust the pixel-to-cm² conversion factor to account for the 10x magnification
# Assuming the original scale without magnification would be 0.001 cm² per pixel
pixel_to_cm2 = 0.002 / (10 * 10)  # Dividing by 100 to account for 10x magnification

@router.post("/callus_identification")
async def predictCallusArea(file: UploadFile = File(...)):
    img = Image.open(BytesIO(await file.read()))
    img_cv2 = np.array(img)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

    results = model(img)

    class_name = "Callus"
    color = (0, 255, 0)

    predictions_metadata = []

    if results[0].masks is not None:
        masks = results[0].masks.data
        confidences = results[0].boxes.conf

        for mask, confidence in zip(masks, confidences):
            mask_np = mask.cpu().numpy()
            mask_np = cv2.resize(mask_np, (img_cv2.shape[1], img_cv2.shape[0]))

            # Calculate the area of the callus in pixels
            callus_area_pixels = np.sum(mask_np > 0.5)

            # Convert pixel area to cm² using the adjusted scale
            callus_area_cm2 = callus_area_pixels * pixel_to_cm2

            img_cv2[mask_np > 0.5] = img_cv2[mask_np > 0.5] * 0.5 + np.array(color) * 0.5

            predictions_metadata.append({
                "label": class_name,
                "confidence": float(confidence),
                "area_pixels": int(callus_area_pixels),
                "area_cm2": callus_area_cm2
            })

    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_cv2)
    buf = BytesIO()
    img_pil.save(buf, format='JPEG')
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/jpeg", headers={
        "X-Predictions": str(predictions_metadata)
    })
