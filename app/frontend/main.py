import os
from fastapi import FastAPI, UploadFile, File
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings


app = FastAPI()


# Configuration
os.environ["GOOGLE_API_KEY"] = "AIzaSyDtBdg1UhMIWyAUMWYiy7oCJahD8lKr1kY"
os.environ["PINECONE_API_KEY"] = "pcsk_4nwGjJ_MGofLEVryLG2WJVNUPRZ59ThTXtZBS3j4LqX6jEVv8dqBuREvfVC7hNx3FLHH2K"
INDEX_NAME = "enterprise-knowledge-base"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    # 1. Save and Load PDF
    with open(f"temp_{file.filename}", "wb") as f:
        f.write(file.file.read())
    
    loader = PyPDFLoader(f"temp_{file.filename}")
    data = loader.load()
    
    # 2. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    
    # 3. Store in Pinecone
    PineconeVectorStore.from_documents(chunks, embeddings, index_name=INDEX_NAME)
    return {"message": "Document ingested and vectorized successfully!"}

@app.get("/ask")
async def ask_question(query: str):
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    
    # 4. Retrieval + Generation (RAG)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    response = qa_chain.invoke(query)
    return {"answer": response["result"]}