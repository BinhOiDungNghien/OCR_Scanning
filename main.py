import os
import torch
import easyocr
import tempfile
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)

def initialize_reader(languages=['en'], gpu=True):
    try:
        reader = easyocr.Reader(languages, gpu=gpu)
        return reader
    except Exception as e:
        app.logger.error(f"Error initializing EasyOCR reader: {e}")
        return None

def extract_text_as_lines(image_path, threshold=10):
    """
    Extracts text from an image and organizes it into lines based on the vertical positions of the text.

    Parameters:
    reader (object): The OCR reader object that will be used to read the text from the image.
    image_path (str): The path to the image file.
    threshold (int, optional): The maximum vertical distance between two text segments to be considered as part of the same line. Defaults to 10.

    Returns:
    list: A list of strings, each representing a line of text extracted from the image.
    """
    try:
        # Check if the image file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Use the OCR reader to extract text from the image
        result = reader.readtext(image_path)

        # Sort the results based on the vertical position (y-coordinate) of the text
        result.sort(key=lambda x: x[0][0][1])

        lines = []
        current_line = []
        current_y = result[0][0][0][1]

        for (bbox, text, prob) in result:
            top_left = bbox[0]
            y = top_left[1]

            # Check if the current text box is within the threshold distance of the current line's y position
            if abs(y - current_y) <= threshold:
                current_line.append((text, bbox[0][0]))
            else:
                # Sort the current line by the x position to ensure left-to-right ordering
                current_line.sort(key=lambda x: x[1])
                # Join the texts in the current line
                lines.append(' '.join([t[0] for t in current_line]))
                # Start a new line
                current_line = [(text, bbox[0][0])]
                current_y = y

        # Append the last line if any
        if current_line:
            current_line.sort(key=lambda x: x[1])
            lines.append(' '.join([t[0] for t in current_line]))

        return lines

    except FileNotFoundError as fnf_error:
        logging.error(fnf_error)
        return []
    except Exception as e:
        logging.error(f"Error extracting text from image: {e}")
        return []

# Check if GPU is available
gpu_available = torch.cuda.is_available()
app.logger.info(f"GPU available: {gpu_available}")

# Initialize the EasyOCR reader with GPU support if available
reader = initialize_reader(['en'], gpu=gpu_available)

@app.route('/')
def home():
    return "Welcome to the OCR API. Use the /extract-text endpoint to upload an image and extract text."

@app.route('/extract-text', methods=['POST'])
def extract_text():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        image = request.files['image']
        
        # Use tempfile to create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            image_path = temp.name
            image.save(image_path)

        lines = extract_text_as_lines(reader, image_path)

        # Remove the temporary file after processing
        os.remove(image_path)

        if lines:
            return jsonify({"lines": lines})
        else:
            return jsonify({"error": "No text extracted"}), 500
    except Exception as e:
        app.logger.error(f"Unhandled exception: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
