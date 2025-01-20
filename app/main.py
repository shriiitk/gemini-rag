import sys
import os
import tempfile
import time
import logging
from gtts import gTTS
# Ensure correct path for importing app components
current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import streamlit as st
from app.components.chat_interface import display_chat_messages, get_user_input
from app.utils.vector_db import load_and_split_documents, initialize_vector_db, perform_similarity_search
from app.utils.gemini_utils import generate_response, build_rag_prompt
from app.audio.audio_processing import transcribe_audio, synthesize_speech, play_audio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        try:
            logging.info(f"Loading data from: {DATA_FILE_PATH}")
            texts = load_and_split_documents(DATA_FILE_PATH)
            if texts:
                st.session_state.vector_db = initialize_vector_db(texts)
                logging.info("Vector database initialized successfully.")
            else:
                st.error("Failed to load document and initialize the vector DB. Please check data file.")
                logging.error(f"Failed to load data from: {DATA_FILE_PATH}")
        except Exception as e:
            st.error(f"An unexpected error occurred during DB initialization: {e}")
            logging.exception("An unexpected error occurred during DB initialization")


# --- Audio Input/Processing ---
if 'recording' not in st.session_state:
    st.session_state.recording = False
audio_data = None

def record_audio():
    import sounddevice as sd
    import soundfile as sf
    global audio_data
    st.session_state.recording = True
    fs = 44100  # Sample rate
    duration = 5  # Seconds
    logging.info(f"Starting audio recording for {duration} seconds...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    st.session_state.recording = False
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file_name = temp_audio_file.name
        sf.write(temp_audio_file_name, audio_data, fs)
        logging.info(f"Audio recording saved to: {temp_audio_file_name}")
        return temp_audio_file_name

def process_audio(temp_audio_file_name):
    with st.spinner("Processing audio..."):
        try:
            user_input = transcribe_audio(temp_audio_file_name)
            if user_input and "Error" not in user_input:
                st.session_state.chat_history.append(("user", user_input))
                logging.info(f"User audio transcribed as: {user_input}")
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
                    # Audio Synthesis
                    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    synthesize_speech(ai_response, temp_audio_file.name)
                    play_audio(temp_audio_file.name)
                    os.remove(temp_audio_file_name)
                    logging.info(f"Audio response generated and played.")
                    st.rerun()
            else:
                st.error(user_input)
                logging.error(f"Transcription failed: {user_input}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logging.exception("An unexpected error occurred during audio processing")


# --- Chat Interface ---
display_chat_messages(st.session_state.chat_history)

# --- Handle Text Input ---
user_input = get_user_input()
if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.spinner("Fetching answer using RAG technique ..."):
        try:
            relevant_documents = perform_similarity_search(st.session_state.vector_db, user_input)
            if relevant_documents:
                context = "\n\n".join(relevant_documents)
                rag_prompt = build_rag_prompt(user_input, context)
                ai_response = generate_response(rag_prompt)
            else:
                ai_response = generate_response(user_input)
            st.session_state.chat_history.append(("ai", ai_response))
            # Audio Synthesis
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            synthesize_speech(ai_response, temp_audio_file.name)
            play_audio(temp_audio_file.name)
            st.rerun()
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logging.exception("An unexpected error occurred during RAG processing")

# --- Audio Input/Processing Button ---
if st.button("Record and Process Audio", key="audio_button"):
    if not st.session_state.recording:
        try:
            temp_audio_file_name = record_audio()
            process_audio(temp_audio_file_name)
        except Exception as e:
            st.error(f"An unexpected error occurred during audio recording: {e}")
            logging.exception("An unexpected error occurred during audio recording")