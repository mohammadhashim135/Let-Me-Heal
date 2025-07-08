from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

def load_pdf_files(directory_path):
    """
    Loads all PDF files from a directory using PyPDFLoader.
    
    Args:
        directory_path (str): Path to the directory containing PDF files.
    
    Returns:
        list: List of loaded documents.
    """
    loader = DirectoryLoader(
        path=directory_path,
        glob="**/*.pdf",  
        loader_cls=PyPDFLoader
    )
    
    documents = loader.load()
    return documents

DATA_PATH = "data"  
documents = load_pdf_files(DATA_PATH)
# print(f"Loaded {len(documents)} documents.")

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)
# print(len(text_chunks))

def get_embedding_model():
    embedding_model =  HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

embedding_model = get_embedding_model()


DB_FAISS_PATH = "vectorstore/db_faiss"
db=FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)