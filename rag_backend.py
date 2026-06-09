import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage

# Global configurations
DB_DIR = os.path.join("data", "chroma_db")

def get_embeddings_instance():
    """Uses Hugging Face's free hosted Inference API instead of running math locally."""
    from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings
    
    # Securely pulls the token you already saved in Render's environment settings
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

def index_resume_text(text_content, filename):
    """Slices text content and logs it into a local Chroma vector database via API embeddings."""
    from langchain_community.vectorstores import Chroma
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(texts=[text_content], metadatas=[{"source": filename}])
    
    embeddings = get_embeddings_instance()
    
    vectorstore = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    return vectorstore

def query_resume_rag(user_question):
    """Retrieves relevant chunks and runs a conversational Hugging Face query."""
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
    
    if not os.path.exists(DB_DIR):
        return "Please upload and index a resume first!"
        
    embeddings = get_embeddings_instance()
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # 1. Fetch relevant resume sections
    docs = retriever.invoke(user_question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 2. Configure a native conversational endpoint model (Llama 3) via API
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