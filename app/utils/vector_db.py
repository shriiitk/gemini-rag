from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS  # Corrected import
import os
from typing import List
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def load_and_split_documents(file_path: str) -> List[str]:
    """Loads and splits documents."""
    try:
        loader = TextLoader(file_path)
        documents = loader.load()
    except Exception as e:
        st.error(f"Error loading document: {e}")
        return []

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    return [doc.page_content for doc in docs]


def initialize_vector_db(texts: List[str]) -> FAISS:
    """Initializes FAISS vector database."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = FAISS.from_texts(texts, embedding=embeddings)
    return vector_db


def perform_similarity_search(vector_db: FAISS, query: str, k: int = 3) -> List[str]:
    """Performs similarity search."""
    if not vector_db:
        st.error("Vector database is not initialized.")
        return []

    docs = vector_db.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]