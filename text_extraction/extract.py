import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_scanned_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))  # Increased resolution for better OCR
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Perform OCR on the image
            page_text = pytesseract.image_to_string(img, config='--psm 6')
            
            # Simple paragraph separation
            paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
            formatted_text = '\n\n'.join(paragraphs)
            
            full_text += f"Page {page_num + 1}:\n{formatted_text}\n\n{'='*50}\n\n"
        
        return full_text
    except Exception as e:
        return f"An error occurred: {str(e)}"

def process_pdf(pdf_path):
    extracted_text = extract_text_from_scanned_pdf(pdf_path)
    return extracted_text

# Usage
pdf_path = 'doc.pdf'  # Make sure this path is correct
if not os.path.exists(pdf_path):
    print(f"Error: The file '{pdf_path}' does not exist.")
else:
    text = process_pdf(pdf_path)
    print("Extracted Text:")
    print(text)