from flask import Flask, request, jsonify  
from ultralytics import YOLO  
import cv2  
import random  
import os  
import time  
import uuid  

# Initialize Flask app  
app = Flask(__name__)  

# Load YOLO models  
model1 = YOLO("./best.pt")  
model2 = YOLO("./yolov8m-seg.pt")  

# Get class names  
yolo_classes1 = list(model1.names.values())  
yolo_classes2 = list(model2.names.values())  
combined_classes = list(set(yolo_classes1) | set(yolo_classes2))  # Combine unique class names  

# Generate random colors for each class  
colors = [random.choices(range(256), k=3) for _ in range(len(combined_classes))]  

# Define confidence threshold  
conf = 0.5  

# Function to process results from each model  
def process_results(results, classes):  
    preds = []  
    for result in results:  
        if result is None:  
            continue  # Skip if the result is None  

        # Check for masks and boxes attributes  
        if not hasattr(result, 'masks') or result.masks is None or len(result.masks) == 0:  
            continue  # Skip if masks are not present or empty  

        if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:  
            continue  # Skip if boxes are not present or empty  

        # Extract predictions if masks and boxes exist  
        for mask, box in zip(result.masks.xy, result.boxes):  
            points = [{"x": float(point[0]), "y": float(point[1])} for point in mask]  
            bbox = box.xywh[0].tolist()  # Center x, y, width, height  
            preds.append({  
                "x": bbox[0],  
                "y": bbox[1],  
                "width": bbox[2],  
                "height": bbox[3],  
                "confidence": float(box.conf[0]),  
                "class": classes[int(box.cls[0])],  
                "points": points,  
                "detection_id": str(uuid.uuid4()),  
            })  
    return preds  

@app.route('/api/v1/get-image-polygon', methods=['POST'])  
def predict():  
    # Check if the file is in the request  
    if 'image' not in request.files:  
        return jsonify({"error": "No image file provided"}), 400  

    # Read the image file  
    file = request.files['image']  
    file_path = "temp.jpg"  
    file.save(file_path)  

    # Read image with OpenCV  
    img = cv2.imread(file_path)  
    height, width, _ = img.shape  

    # Prepare a list to hold predictions from both models  
    predictions = []  

    # Predict with the first YOLO model  
    start_time = time.time()  
    results1 = model1.predict(img, conf=conf)  
    inference_time1 = time.time() - start_time  

    # Process predictions from the first model  
    predictions.extend(process_results(results1, yolo_classes1))  

    # Predict with the second YOLO model  
    start_time = time.time()  
    results2 = model2.predict(img, conf=conf)  
    inference_time2 = time.time() - start_time  

    # Process predictions from the second model  
    predictions.extend(process_results(results2, yolo_classes2))  

    # Generate response in the specified schema  
    response = {  
        "detection_id": str(uuid.uuid4()),  
        "time": inference_time1 + inference_time2,  # Total inference time  
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