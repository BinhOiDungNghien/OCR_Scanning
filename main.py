import os
import torch
import easyocr
from flask import Flask, request, jsonify

app = Flask(__name__)

def initialize_reader(languages=['en'], gpu=True):
    try:
        reader = easyocr.Reader(languages, gpu=gpu)
        return reader
    except Exception as e:
        print(f"Error initializing EasyOCR reader: {e}")
        return None

def extract_text_as_lines(reader, image_path, threshold=10):
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        result = reader.readtext(image_path)
        result.sort(key=lambda x: x[0][0][1])

        lines = []
        current_line = []
        current_y = result[0][0][0][1]

        for (bbox, text, prob) in result:
            top_left = bbox[0]
            y = top_left[1]
            if abs(y - current_y) > threshold:
                lines.append(' '.join(current_line))
                current_line = [text]
                current_y = y
            else:
                current_line.append(text)

        if current_line:
            lines.append(' '.join(current_line))

        return lines

    except FileNotFoundError as fnf_error:
        print(fnf_error)
        return []
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return []

# Check if GPU is available
gpu_available = torch.cuda.is_available()
print(f"GPU available: {gpu_available}")

# Initialize the EasyOCR reader with GPU support if available
reader = initialize_reader(['en'], gpu=gpu_available)

@app.route('/extract-text', methods=['POST'])
def extract_text():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']
    image_path = os.path.join('/tmp', image.filename)
    image.save(image_path)

    lines = extract_text_as_lines(reader, image_path)

    if lines:
        return jsonify({"lines": lines})
    else:
        return jsonify({"error": "No text extracted"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
