import os
from fastapi import APIRouter, UploadFile, File, Response
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

router = APIRouter()

model_path = os.path.join(os.path.dirname(__file__), '../models/plant_model.pt')
model = YOLO(model_path)

@router.post("/plant_status_identification")
async def predictPlantHealth(file: UploadFile = File(...)):
    img = Image.open(BytesIO(await file.read()))
    img_cv2 = np.array(img)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)

    results = model(img)

    class_names = {0: 'Browning', 1: 'Flask'}
    colors = {
        'Browning': (0, 255, 0),
        'Flask': (0, 0, 255)
    }

    predictions_metadata = []

    for result in results[0].boxes:
        label_index = int(result.cls)
        confidence = float(result.conf)
        bbox = result.xyxy.tolist()

        label = class_names[label_index]
        color = colors[label]

        predictions_metadata.append({
            "label": label,
            "confidence": confidence
        })

        for box in bbox:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_cv2, (x1, y1), (x2, y2), color, 2)
            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(img_cv2, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_cv2)
    buf = BytesIO()
    img_pil.save(buf, format='JPEG')
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/jpeg", headers={
        "X-Predictions": str(predictions_metadata)
    })
