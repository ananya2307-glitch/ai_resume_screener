import os
from groq import Groq

# Stores uploaded resume text in memory
RESUME_TEXT = ""


def index_resume_text(text_content, filename):
    global RESUME_TEXT

    RESUME_TEXT = text_content

    return True


def query_resume_rag(user_question):
    global RESUME_TEXT

    if not RESUME_TEXT.strip():
        return "Please upload a resume first."

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "GROQ_API_KEY not configured."

    try:
        client = Groq(api_key=api_key)

        prompt = f"""
You are an AI Resume Screening Assistant.

Use ONLY the information present in the resume.

If the answer cannot be found in the resume, respond with:
'I could not find that information in the resume.'

Resume:
{RESUME_TEXT[:12000]}

Question:
{user_question}
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert resume analysis assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=500
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"