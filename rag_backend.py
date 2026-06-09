import os

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from groq import Groq


# =========================
# CHROMA DATABASE
# =========================

CHROMA_PATH = "chroma_db"

client_db = PersistentClient(path=CHROMA_PATH)

collection = client_db.get_or_create_collection(
    name="resume_collection"
)


# =========================
# EMBEDDING MODEL
# =========================

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# =========================
# INDEX RESUME
# =========================

def index_resume_text(text_content, filename):

    chunks = []

    chunk_size = 500
    overlap = 100

    start = 0

    while start < len(text_content):

        end = start + chunk_size

        chunks.append(
            text_content[start:end]
        )

        start += chunk_size - overlap

    embeddings = embedding_model.encode(
        chunks
    ).tolist()

    ids = [
        f"{filename}_{i}"
        for i in range(len(chunks))
    ]

    try:

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=[
                {"source": filename}
                for _ in chunks
            ]
        )

    except Exception:
        pass

    return True


# =========================
# QUERY RAG
# =========================

def query_resume_rag(user_question):

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:

        return (
            "GROQ_API_KEY is missing. "
            "Add it inside Render Environment Variables."
        )

    question_embedding = embedding_model.encode(
        user_question
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    retrieved_docs = results["documents"][0]

    if len(retrieved_docs) == 0:

        return "Please upload a resume first."

    context = "\n\n".join(
        retrieved_docs
    )

    client = Groq(
        api_key=api_key
    )

    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        temperature=0.2,
        max_tokens=500,
        messages=[
            {
                "role": "system",
                "content": f"""
You are an AI Resume Screener.

Answer ONLY from the resume context.

If the answer is not available in the resume,
say:

'I could not find that information in the resume.'

Resume Context:

{context}
"""
            },
            {
                "role": "user",
                "content": user_question
            }
        ]
    )

    return completion.choices[0].message.content