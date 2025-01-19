from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from typing import List
import streamlit as st
from dotenv import load_dotenv
import chromadb
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def load_and_split_documents(file_path: str) -> List[str]:
    """
    Loads a text file and splits it into chunks for vectorization.

    Args:
        file_path (str): Path to the text file.

    Returns:
        List[str]: A list of text chunks.
    """
    try:
        loader = TextLoader(file_path)
        documents = loader.load()
    except Exception as e:
        st.error(f"Error loading document: {e}")
        return []

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    return [doc.page_content for doc in docs]


def initialize_vector_db(texts: List[str]) -> Chroma:
    """
    Initializes and populates a Chroma vector database.

    Args:
        texts (List[str]): A list of text strings to be embedded.

    Returns:
        Chroma: An initialized Chroma vector database instance.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    client = chromadb.Client(settings=chromadb.Settings(is_persistent=False))
    vector_db = Chroma(embedding_function=embeddings, client=client, collection_name="my_collection")
    vector_db.add_texts(texts)
    return vector_db


def perform_similarity_search(vector_db: Chroma, query: str, k: int = 3) -> List[str]:
    """
    Performs a similarity search in the vector database.

    Args:
        vector_db (Chroma): The Chroma vector database instance.
        query (str): The query string to search for.
        k (int): The number of nearest neighbors to return.

    Returns:
        List[str]: A list of text chunks that are the most similar to the query.
    """
    if not vector_db:
        st.error("Vector database is not initialized.")
        return []

    retriever = vector_db.as_retriever(search_kwargs={"k": k})
    relevant_documents = retriever.get_relevant_documents(query)
    return [doc.page_content for doc in relevant_documents]