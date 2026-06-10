from dotenv import load_dotenv
load_dotenv()
import os
import pypdf

from flask import Flask, render_template, request, jsonify

from rag_backend import (
    index_resume_text,
    query_resume_rag
)

app = Flask(__name__)

UPLOAD_FOLDER = "data"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():

    if "file" not in request.files:
        return jsonify({
            "error": "No file uploaded"
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "No file selected"
        }), 400

    file_path = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(file_path)

    try:

        text_content = ""

        if file.filename.lower().endswith(".pdf"):

            reader = pypdf.PdfReader(
                file_path
            )

            for page in reader.pages:
                text_content += (
                    page.extract_text() or ""
                )

        elif file.filename.lower().endswith(".txt"):

            with open(
                file_path,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:

                text_content = f.read()

        else:

            return jsonify({
                "error":
                "Only PDF and TXT files are supported."
            }), 400

        if not text_content.strip():

            return jsonify({
                "error":
                "No readable text found."
            }), 400

        index_resume_text(
            text_content,
            file.filename
        )

        return jsonify({
            "message":
            f"{file.filename} uploaded successfully."
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


@app.route("/query", methods=["POST"])
def query():

    try:

        data = request.get_json()

        question = data.get(
            "question",
            ""
        ).strip()

        if not question:

            return jsonify({
                "answer":
                "Please enter a question."
            }), 400

        answer = query_resume_rag(
            question
        )

        return jsonify({
            "answer": answer
        })

    except Exception as e:

        return jsonify({
            "answer": str(e)
        }), 500


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8080
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )