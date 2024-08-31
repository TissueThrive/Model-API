import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Response
from io import BytesIO
from PIL import Image

app = FastAPI()

def color_based_segmentation(image_data):
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image_data, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the HSV values to detect the callus
    lower_hsv = np.array([10, 40, 40])  # Example values, adjust based on your image
    upper_hsv = np.array([30, 255, 255])

    # Create a mask with the defined bounds
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Optional: Apply some morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Apply the mask to the original image to get the segmented callus
    segmented_image = cv2.bitwise_and(image_data, image_data, mask=mask)

    return segmented_image

@app.post("/segment/")
async def predictCallustColor(file: UploadFile = File(...)):
    # Read the image file
    image = Image.open(BytesIO(await file.read()))
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Perform color-based segmentation
    segmented_image = color_based_segmentation(image_np)

    # Convert the segmented image back to RGB for saving
    segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    segmented_pil = Image.fromarray(segmented_image_rgb)
    buf = BytesIO()
    segmented_pil.save(buf, format='JPEG')
    buf.seek(0)

    return Response(content=buf.getvalue(), media_type="image/jpeg")


