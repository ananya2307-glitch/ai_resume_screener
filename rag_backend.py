import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# Safe, serverless API embeddings using the class Render expected
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction"
)

def process_resume(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

def query_resume(vector_store, question):
    docs = vector_store.similarity_search(question, k=2)
    context = " ".join([doc.page_content for doc in docs])
    return context