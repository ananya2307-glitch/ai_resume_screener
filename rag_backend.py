import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceInferenceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage

# Global configurations
DB_DIR = os.path.join("data", "chroma_db")

# Initialize the embedding model locally
embeddings = HuggingFaceInferenceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
def index_resume_text(text_content, filename):
    """Slices text content and logs it into a local Chroma vector database."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(texts=[text_content], metadatas=[{"source": filename}])
    
    # Creation syntax uses 'embedding'
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    return vectorstore

def query_resume_rag(user_question):
    """Retrieves relevant chunks and runs a conversational Hugging Face query."""
    if not os.path.exists(DB_DIR):
        return "Please upload and index a resume first!"
        
    # Loading syntax strictly uses 'embedding_function'
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # 1. Fetch relevant resume sections
    docs = retriever.invoke(user_question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 2. Configure a native conversational endpoint model (Llama 3)
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        temperature=0.1,
        max_new_tokens=512
    )
    
    # 3. Use ChatHuggingFace wrapper safely
    chat_model = ChatHuggingFace(llm=llm)
    
    # 4. Construct structural system and user roles
    messages = [
        SystemMessage(content=f"You are a helpful HR robot. Answer using only the resume facts given below. If the information is not present, say you cannot find it.\n\nContext:\n{context}"),
        HumanMessage(content=user_question)
    ]
    
    # 5. Execute generation
    response = chat_model.invoke(messages)
    return response.content