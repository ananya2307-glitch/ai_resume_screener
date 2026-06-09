import pypdf
import pandas as pd
import os

def extract_text_from_pdf(pdf_path):
    # This reads the words out loud from the PDF file
    text = ""
    with open(pdf_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text

def log_resume_metadata(filename, text):
    # This uses Pandas to capture core word counts and key skills
    word_count = len(text.split())
    
    # We tell the computer to look out for these specific tech skills!
    keywords = ["python", "javascript", "aws", "excel", "sql"]
    found_skills = [skill for skill in keywords if skill in text.lower()]
    
    new_data = {
        "Filename": [filename],
        "WordCount": [word_count],
        "KeySkillsDetected": [", ".join(found_skills)]
    }
    return new_data