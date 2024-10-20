from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image
import io
import numpy as np
import torch
from ultralytics import YOLO
import os

app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:8000"}})
CORS(app, resources={r"/*": {"origins": "*"}})  # Change the origin to '*' to allow all origins or set it specifically to 'http://localhost:8000'

# Load YOLO model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# for mac users
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = YOLO('./FastSAM.pt')

# Directory to save segmented images
OUTPUT_DIR = 'segmented_images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Utility functions
def resize_image(image, input_size):
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))
    return image

def format_results(result, filter=0):
    annotations = []
    n = len(result.masks.data)
    for i in range(n):
        annotation = {}
        mask = result.masks.data[i] == 1.0

        if torch.sum(mask) < filter:
            continue
        annotation["id"] = i
        annotation["segmentation"] = mask.cpu().numpy()
        annotation["bbox"] = result.boxes.data[i]
        annotation["score"] = result.boxes.conf[i]
        annotation["area"] = annotation["segmentation"].sum()
        annotations.append(annotation)
    return annotations

def point_prompt(masks, points, point_label):
    h = masks[0]["segmentation"].shape[0]
    w = masks[0]["segmentation"].shape[1]

    onemask = np.zeros((h, w))
    masks = sorted(masks, key=lambda x: x['area'], reverse=False)

    for i, annotation in enumerate(masks):
        if type(annotation) == dict:
            mask = annotation['segmentation']
        else:
            mask = annotation

        for j, point in enumerate(points):
            # Convert points to integer coordinates
            point_x = int(point[0])
            point_y = int(point[1])

            if mask[point_y, point_x] == 1:
                if point_label[j] == 1:
                    onemask[mask] = 1
                elif point_label[j] == 0:
                    onemask[mask] = 0
    onemask = onemask >= 1
    return onemask, 0

def extract_selected_object(image, mask):
    # Convert image to RGBA format
    image = image.convert("RGBA")
    mask_uint8 = mask.astype(np.uint8) * 255

    # Create a blank alpha channel image
    masked_image_array = np.array(image)
    masked_image_array[..., 3] = mask_uint8  # Update alpha channel based on mask

    # Convert back to PIL Image
    masked_image = Image.fromarray(masked_image_array)

    return masked_image

@app.route('/resize', methods=['POST'])
def resize_uploaded_image():
    if 'image' not in request.files:
        return "No image provided", 400

    file = request.files['image']

    # Load the image and resize it
    raw_image = Image.open(file.stream)
    resized_image = resize_image(raw_image, input_size=1024)

    # Convert the resized image to bytes for returning
    img_byte_arr = io.BytesIO()
    resized_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return send_file(img_byte_arr, mimetype='image/png')

@app.route('/segment', methods=['POST'])
def segment_image():
    # Ensure the image is received correctly
    if 'image' not in request.files:
        return "No image provided", 400

    # Get data from request
    file = request.files['image']
    points = request.form.get('points')
    labels = request.form.get('labels')

    # Convert points and labels from string to list of tuples/integers
    points = eval(points)
    labels = eval(labels)

    # Load and process image
    raw_image = Image.open(file.stream)
    resized_image = resize_image(raw_image, input_size=1024)

    # Run YOLO model
    results = model(resized_image, device=device, retina_masks=True)
    results = format_results(results[0], 0)

    # Segment using provided points and labels
    masks, _ = point_prompt(results, points, labels)

    # Extract selected object from image using mask
    selected_object_image = extract_selected_object(resized_image, masks)

    # Save segmented object as PNG
    output_filename = os.path.join(OUTPUT_DIR, 'selected_object.png')
    selected_object_image.save(output_filename, format='PNG')

    # Convert PIL image to bytes for returning to client
    img_byte_arr = io.BytesIO()
    selected_object_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # Return the image bytes and download URL
    return send_file(img_byte_arr, mimetype='image/png', as_attachment=False, download_name='selected_object.png')

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    return send_file(os.path.join(OUTPUT_DIR, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
