import os
from fastapi import APIRouter, UploadFile, File, Response
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

router = APIRouter()

model_path = os.path.join(os.path.dirname(__file__), '../models/callus_shape_model.pt')
model = YOLO(model_path)

@router.post("/callus_shape_identification")
async def predictCallusShape(file: UploadFile = File(...)):
    img = Image.open(BytesIO(await file.read()))
    img_cv2 = np.array(img)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

    results = model(img)

    class_names = ["Ear-shape", "Frilly-shape"]
    colors = {
        "Ear-shape": (0, 255, 0),
        "Frilly-shape": (0, 0, 255)
    }

    predictions_metadata = []

    if results[0].masks is not None:
        masks = results[0].masks.data
        confidences = results[0].boxes.conf
        classes = results[0].boxes.cls

        for mask, confidence, cls in zip(masks, confidences, classes):
            mask_np = mask.cpu().numpy()
            mask_np = cv2.resize(mask_np, (img_cv2.shape[1], img_cv2.shape[0]))

            class_name = class_names[int(cls)]
            color = colors[class_name]

            img_cv2[mask_np > 0.5] = img_cv2[mask_np > 0.5] * 0.5 + np.array(color) * 0.5

            predictions_metadata.append({
                "label": class_name,
                "confidence": float(confidence)
            })

    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_cv2)
    buf = BytesIO()
    img_pil.save(buf, format='JPEG')
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/jpeg", headers={
        "X-Predictions": str(predictions_metadata)
    })
