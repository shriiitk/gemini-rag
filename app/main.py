import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from app.components.chat_interface import display_chat_messages, get_user_input
from app.utils.vector_db import load_and_split_documents, initialize_vector_db, perform_similarity_search
from app.utils.gemini_utils import generate_response, build_rag_prompt
import os


# --- App Configuration ---
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)
st.title("RAG Chatbot")

# --- Initialize Session State ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None

DATA_FILE_PATH = "data/HoroscopeBook.txt"

# --- Load Data and Setup Vector DB ---
if not st.session_state.vector_db:
    with st.spinner("Initializing the Vector database ..."):
        texts = load_and_split_documents(DATA_FILE_PATH)
        if texts:
            st.session_state.vector_db = initialize_vector_db(texts)
        else:
            st.error("Failed to load document and initialize the vector DB. Please check data file.")


# --- Chat Interface ---
display_chat_messages(st.session_state.chat_history)

user_input = get_user_input()

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.spinner("Fetching answer using RAG technique ..."):
        # Perform Similarity Search
        relevant_documents = perform_similarity_search(st.session_state.vector_db, user_input)
        if relevant_documents:
            context = "\n\n".join(relevant_documents)
            rag_prompt = build_rag_prompt(user_input, context)
            ai_response = generate_response(rag_prompt)
        else:
            ai_response = generate_response(user_input)
    st.session_state.chat_history.append(("ai", ai_response))
    st.rerun()