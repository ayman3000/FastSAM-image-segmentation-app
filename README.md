# Image Segmentation Application

This repository contains an image segmentation application built using a Flask backend and a JavaScript-powered HTML frontend. The application allows users to upload an image, mark specific regions for segmentation, and download the segmented result. The system leverages the YOLO model for segmentation.

## Overview

This project is a direct implementation of the FastSAM Model from Meta, specifically the "SAM: Segment Anything Model." It utilizes the `FastSAM.pt` model for segmenting various regions within an image based on user inputs.

The application consists of two main components:

1. **Backend (Flask API)**: Handles image resizing, point-based segmentation, and returns segmented images.
2. **Frontend (HTML & JavaScript)**: Allows users to upload images, add segmentation points, interact with the backend API, and view/download the segmented results.

## Features

- **Upload Images**: Users can upload images for processing.
- **Add Segmentation Points**: Users can mark positive and negative points to guide the segmentation process.
- **Automatic Segmentation**: After adding points, the system segments the image based on the selected points.
- **Download Segmented Object**: The segmented portion of the image can be downloaded.

## Project Structure

- `app.py`: The Flask application that serves the backend API for image processing.
- `index.html`: The front-end interface that allows users to interact with the segmentation application.
- `static/` and `templates/`: Directories containing static assets and HTML templates.

## Prerequisites

- Python 3.7 or higher
- Flask
- Torch (PyTorch for model inference)
- PIL (Pillow for image manipulation)
- JavaScript-enabled web browser

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/image-segmentation-app.git
   cd image-segmentation-app
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```
   The server will start at `http://127.0.0.1:5000/` by default.

2. **Open the HTML frontend**:
   - Navigate to `index.html` in a web browser (e.g., by serving it with a local server).

## Application Flow

### Backend (`app.py`)

The Flask API provides two main endpoints:

1. **`/resize` [POST]**: Resizes an uploaded image to a fixed size for processing.
   - **Input**: Image file
   - **Output**: Resized image

2. **`/segment` [POST]**: Segments the image based on user-specified points (positive or negative).
   - **Input**: Image, list of points, and their corresponding labels
   - **Output**: Segmented image with the selected region highlighted

### Frontend (`index.html`)

The HTML page includes:
- An image upload feature for users to submit an image for processing.
- Click handlers to add positive (green) or negative (red) points for segmentation.
- A button to clear all added points.
- JavaScript logic to communicate with the Flask API, send requests, and update the UI accordingly.

## Usage

1. **Upload an Image**: Click on the "Choose File" button to upload an image.
2. **Add Segmentation Points**:
   - Click on the image to add positive points (green markers).
   - Hold `Shift` and click to add negative points (red markers).
3. **Automatic Segmentation**: Each click triggers a segmentation process.
4. **Clear Points**: Click "Remove All Points" to clear all markers.
5. **Download Segmented Object**: After segmentation, download the result using the provided link.

## Technical Details

- **FastSAM Model by Meta**: The application directly implements the FastSAM model from Meta, which is part of the "Segment Anything Model" (SAM) series designed to perform general-purpose segmentation tasks. The model (`FastSAM.pt`) is used for the segmentation tasks on the uploaded images.

- **YOLO Model**: The application uses a YOLO-based model (`FastSAM.pt`) to perform segmentation tasks on the uploaded images.

- **Image Processing**: The application utilizes `Pillow` to resize and manipulate images, and `NumPy` to process mask data.

- **CORS Configuration**: The Flask app is configured to handle CORS issues, allowing requests from the frontend to communicate with the backend.

## File Descriptions

### `app.py`

- Contains the backend logic for image resizing and segmentation.
- Key functions:
  - **`resize_image`**: Resizes an image to a specified input size while maintaining the aspect ratio.
  - **`format_results`**: Extracts useful information (e.g., masks, bounding boxes) from the YOLO model's output.
  - **`point_prompt`**: Updates the segmentation mask based on user-provided positive and negative points.
  - **`extract_selected_object`**: Generates a segmented image using the specified mask.

### `index.html`

- Provides the user interface for image upload, point selection, and result visualization.
- Uses JavaScript to handle user interactions, including adding markers and communicating with the backend.
- Features buttons for clearing markers and downloading the segmented image.

## Troubleshooting

- **CORS Errors**: Ensure that the Flask CORS configuration allows the correct origin, especially if accessing from different URLs or ports.
- **Model File (`FastSAM.pt`) Missing**: Ensure the YOLO model file is downloaded and available in the root directory.
- **Backend Not Running**: Make sure to start the Flask server before interacting with the frontend.

## Future Improvements

- **User Authentication**: Add authentication to prevent unauthorized usage of the segmentation service.
- **Model Upgrades**: Implement additional or more sophisticated models for better segmentation quality.
- **Frontend Enhancements**: Improve the user interface for a more seamless user experience, including better marker visualization.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Contact

For any inquiries, please contact [ayman3000@gmail.com].

