import os
import pypdf
from flask import Flask, render_template, request, jsonify
from rag_backend import index_resume_text, query_resume_rag

app = Flask(__name__)
UPLOAD_FOLDER = "data"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
        
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        try:
            # Safely handle PDF vs Text file extraction rules
            if file.filename.lower().endswith('.pdf'):
                reader = pypdf.PdfReader(file_path)
                text_content = ""
                for page in reader.pages:
                    text_content += page.extract_text() or ""
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text_content = f.read()
            
            if not text_content.strip():
                return jsonify({"error": "The uploaded file appears to be empty or unreadable."}), 400
                
            # Stream extracted content directly into your Chroma vector storage
            index_resume_text(text_content, file.filename)
            return jsonify({"message": f"Successfully indexed {file.filename}!"})
            
        except Exception as e:
            return jsonify({"error": f"Failed to process file: {str(e)}"}), 500

@app.route("/query", methods=["POST"])
def query_bot():
    data = request.json
    user_question = data.get("question")
    
    if not user_question:
        return jsonify({"answer": "Please ask a valid question!"}), 400
        
    try:
        # Route query through the updated LangChain conversation stream
        ai_answer = query_resume_rag(user_question)
        return jsonify({"answer": ai_answer})
    except Exception as e:
        return jsonify({"answer": f"Error running AI model backend: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8080)