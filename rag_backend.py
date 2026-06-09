import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage

# Global in-memory storage variable to hold our database instance across requests
GLOBAL_VECTORSTORE = None

def get_embeddings_instance():
    """Uses Hugging Face's hosted community Inference API wrapper for rapid execution."""
    from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
    
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def index_resume_text(text_content, filename):
    """Slices text and builds an instant In-Memory Chroma DB, completely avoiding disk lag."""
    global GLOBAL_VECTORSTORE
    from langchain_community.vectorstores import Chroma
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(texts=[text_content], metadatas=[{"source": filename}])
    
    embeddings = get_embeddings_instance()
    
    # FIX: Initialize Chroma purely in-memory (no persist_directory argument)
    GLOBAL_VECTORSTORE = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings
    )
    return GLOBAL_VECTORSTORE

def query_resume_rag(user_question):
    """Queries the globally held in-memory database instance instantly."""
    global GLOBAL_VECTORSTORE
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
    
    # Check if a file has been processed in memory yet
    if GLOBAL_VECTORSTORE is None:
        return "Please upload and index a resume first!"
        
    retriever = GLOBAL_VECTORSTORE.as_retriever(search_kwargs={"k": 2})
    
    # 1. Fetch relevant sections from memory
    docs = retriever.invoke(user_question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 2. Configure conversational Llama 3 via API
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        temperature=0.1,
        max_new_tokens=512,
        huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    )
    
    chat_model = ChatHuggingFace(llm=llm)
    
    messages = [
        SystemMessage(content=f"You are a helpful HR robot. Answer using only the resume facts given below. If the information is not present, say you cannot find it.\n\nContext:\n{context}"),
        HumanMessage(content=user_question)
    ]
    
    response = chat_model.invoke(messages)
    return response.content