import easyocr
import os
import torch

def initialize_reader(languages=['en'], gpu=True):
    """
    Initialize the EasyOCR reader with specified languages.
    """
    try:
        reader = easyocr.Reader(languages, gpu=gpu)
        return reader
    except Exception as e:
        print(f"Error initializing EasyOCR reader: {e}")
        return None

def extract_text_as_lines(reader, image_path, threshold=10):
    """
    Extract text from an image and organize it into lines.

    Parameters:
    - reader: EasyOCR reader object
    - image_path: Path to the image file
    - threshold: Pixel threshold to detect new lines

    Returns:
    - List of strings, each representing a line of text
    """
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        result = reader.readtext(image_path)
        
        # Sort the results based on the vertical position of the bounding boxes
        result.sort(key=lambda x: x[0][0][1])
        
        lines = []
        current_line = []
        current_y = result[0][0][0][1]  # Initial y position

        for (bbox, text, prob) in result:
            top_left = bbox[0]
            y = top_left[1]
            
            # Check if the text is on a new line
            if abs(y - current_y) > threshold:
                lines.append(' '.join(current_line))
                current_line = [text]
                current_y = y
            else:
                current_line.append(text)
        
        # Append the last line
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines

    except FileNotFoundError as fnf_error:
        print(fnf_error)
        return []
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return []

def main():
    # Check if GPU is available
    gpu_available = torch.cuda.is_available()
    print(f"GPU available: {gpu_available}")

    # Initialize the EasyOCR reader with GPU support if available
    reader = initialize_reader(['en'], gpu=gpu_available)

    if reader is None:
        print("Failed to initialize the EasyOCR reader. Exiting.")
        return

    # Path to the image file
    image_path = 'C:/Users/PLC/Koolsoft/OCR/De-Thi-Mon-Tieng-Anh.jpg'

    # Extract text as lines from the image
    lines = extract_text_as_lines(reader, image_path)

    # Display the results
    if lines:
        print("Extracted text:")
        print(lines)
    else:
        print("No text extracted.")

if __name__ == "__main__":
    main()
