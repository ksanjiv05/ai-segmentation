# from flask import Flask  

# app = Flask(__name__)  

# @app.route('/')  
# def hello_world():  
#     return 'Hello, World!'  

# if __name__ == '__main__':  
#     app.run(debug=True)
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import random
import os
import time
import uuid

# Initialize Flask app
app = Flask(__name__)

# Load YOLO model
model = YOLO("./best.pt")

yolo_classes = list(model.names.values())
classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

# Generate random colors for each class
colors = [random.choices(range(256), k=3) for _ in classes_ids]

# Define confidence threshold
conf = 0.5

@app.route('/api/v1/get-image-polygon', methods=['POST'])
def predict():
    # Check if file is in the request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    # Read the image file
    file = request.files['image']
    file_path = "temp.jpg"
    file.save(file_path)

    # Read image with OpenCV
    img = cv2.imread(file_path)
    height, width, _ = img.shape

    # Predict with YOLO
    start_time = time.time()
    results = model.predict(img, conf=conf)
    inference_time = time.time() - start_time

    # Prepare results
    predictions = []
    for result in results:
        for mask, box in zip(result.masks.xy, result.boxes):
            # points = np.int32([mask]).tolist()  # Mask points as list
            points = [{"x": float(point[0]), "y": float(point[1])} for point in mask]  
            bbox = box.xywh[0].tolist()  # Center x, y, width, height
            predictions.append({
                "x": bbox[0],
                "y": bbox[1],
                "width": bbox[2],
                "height": bbox[3],
                "confidence": float(box.conf[0]),
                "class": yolo_classes[int(box.cls[0])],
                "points": points,
                "detection_id": str(uuid.uuid4()),
            })

    # Generate response in the specified schema
    response = {
        "detection_id": str(uuid.uuid4()),
        "time": inference_time,
        "image": {
            "width": width,
            "height": height
        },
        "predictions": predictions
    }

    # Clean up the temporary file
    os.remove(file_path)

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)