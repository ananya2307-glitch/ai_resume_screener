import os
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage

GLOBAL_VECTORSTORE = None

class DirectHuggingFaceEmbeddings:
    """Manually routes vector extraction directly through a standard web request pipeline."""
    def __init__(self):
        self.token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        self.api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def embed_documents(self, texts):
        response = requests.post(
            self.api_url, 
            headers=self.headers, 
            json={"inputs": texts, "options": {"wait_for_model": True}}
        )
        if response.status_code != 200:
            raise Exception(f"Hugging Face API Error: {response.text}")
        return response.json()

    def embed_query(self, text):
        response = requests.post(
            self.api_url, 
            headers=self.headers, 
            json={"inputs": [text], "options": {"wait_for_model": True}}
        )
        if response.status_code != 200:
            raise Exception(f"Hugging Face API Error: {response.text}")
        return response.json()[0]


def index_resume_text(text_content, filename):
    """Slices text and logs it directly into an In-Memory Chroma instance with telemetry disabled."""
    global GLOBAL_VECTORSTORE
    from langchain_community.vectorstores import Chroma
    from chromadb.config import Settings
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(texts=[text_content], metadatas=[{"source": filename}])
    
    embeddings = DirectHuggingFaceEmbeddings()
    
    # FIX: Explicitly disable Chroma telemetry to bypass the internal argument crash
    chroma_settings = Settings(anonymized_telemetry=False)
    
    GLOBAL_VECTORSTORE = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        client_settings=chroma_settings
    )
    return GLOBAL_VECTORSTORE


def query_resume_rag(user_question):
    """Queries the globally tracked in-memory database instance instantly."""
    global GLOBAL_VECTORSTORE
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
    
    if GLOBAL_VECTORSTORE is None:
        return "Please upload and index a resume first!"
        
    retriever = GLOBAL_VECTORSTORE.as_retriever(search_kwargs={"k": 2})
    
    # 1. Fetch relevant sections
    docs = retriever.invoke(user_question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # 2. Setup structural conversational LLM pipeline
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