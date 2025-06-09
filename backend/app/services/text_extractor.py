import os
from pathlib import Path
import pytesseract
from PIL import Image
import PyPDF2 

def extract_from_pdf(file_path):
    text_pages = [] 
    with open(file_path,'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for i , page in enumerate(reader.pages):
            text = page.extract_text()
            text_pages.append({'page_num':i+1,'text':text})
    return text_pages

def extract_from_image(file_path):
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    return [{'page_num':1,'text':text}]

def extract_from_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return [{'page_num':1,'text':text}]

def extract_text(doc_path):
    ext = os.path.splitext(doc_path)[1].lower()

    if ext == '.pdf':
        return extract_from_pdf(doc_path)

    elif ext in ['.jpg','.png','.jpeg']:
        return extract_from_image(doc_path)

    elif ext == '.txt':
        return extract_from_text(doc_path)

    else:
        raise ValueError(f"Unsupported File Type: {ext}")

    