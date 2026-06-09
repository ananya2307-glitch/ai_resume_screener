import os
from flask import Flask, render_template, request, jsonify
from rag_backend import process_resume, query_resume

app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = './tmp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variable to hold our vector store
current_vector_store = None

@app.route('/')
def home():
    # This serves your index.html from the /templates folder
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global current_vector_store
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        try:
            current_vector_store = process_resume(file_path)
            return jsonify({"message": f"Successfully processed {file.filename}!"}), 200
        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/query', methods=['POST'])
def ask_question():
    global current_vector_store
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
        
    if current_vector_store is None:
        return jsonify({"error": "Please upload a resume first"}), 400
        
    try:
        answer = query_resume(current_vector_store, question)
        return jsonify({"context": answer}), 200
    except Exception as e:
        return jsonify({"error": f"Query failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)