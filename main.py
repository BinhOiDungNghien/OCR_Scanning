from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import easyocr
import os
import torch
from typing import List
import uvicorn

app = FastAPI()

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

gpu_available = torch.cuda.is_available()
print(f"GPU available: {gpu_available}")

reader = initialize_reader(['en'], gpu=gpu_available)

if reader is None:
    raise RuntimeError("Failed to initialize the EasyOCR reader.")

@app.post("/extract-text/")
async def extract_text(file: UploadFile = File(...), threshold: int = 10):
    try:
        file_location = f"./{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())
        
        lines = extract_text_as_lines(reader, file_location, threshold)
        
        os.remove(file_location)

        if not lines:
            return JSONResponse(status_code=200, content={"message": "No text extracted."})
        
        return JSONResponse(status_code=200, content={"lines": lines})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
