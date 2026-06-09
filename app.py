import os
from flask import Flask, request, jsonify
from rag_backend import process_resume, query_resume  # Cleaned to match exact function names

app = Flask(__name__)

UPLOAD_FOLDER = '/tmp'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return jsonify({"status": "healthy", "message": "AI Resume Screener Backend API is Live!"})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file chunk found in request"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        try:
            global current_vector_store
            current_vector_store = process_resume(file_path)
            return jsonify({"message": f"Successfully processed {file.filename}!"}), 200
        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/query', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "No question string provided"}), 400
        
    try:
        if 'current_vector_store' not in globals():
            return jsonify({"error": "Please upload and process a resume profile first"}), 400
            
        context_reply = query_resume(current_vector_store, question)
        return jsonify({"context": context_reply}), 200
    except Exception as e:
        return jsonify({"error": f"Query execution failed: {str(e)}"}), 500

# Hardcoded strictly to 8080 
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)