import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from app.components.chat_interface import display_chat_messages, get_user_input
from app.utils.vector_db import load_and_split_documents, initialize_vector_db, perform_similarity_search
from app.utils.gemini_utils import generate_response, build_rag_prompt
from app.audio.audio_processing import transcribe_audio, synthesize_speech, play_audio
import tempfile
import sounddevice as sd
import soundfile as sf
import streamlit as st
import tempfile
import time
from gtts import gTTS
import logging

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
        texts = load_and_split_documents(DATA_FILE_PATH)
        if texts:
            st.session_state.vector_db = initialize_vector_db(texts)
        else:
            st.error("Failed to load document and initialize the vector DB. Please check data file.")

# Conditional imports for sounddevice and soundfile
if "streamlit" not in sys.modules:
    import sounddevice as sd
    import soundfile as sf

# --- Audio Input/Processing ---
recording = False
audio_data = None

def record_audio():
    global recording, audio_data
    recording = True
    fs = 44100  # Sample rate
    duration = 5  # Seconds
    print(f"Recording audio for {duration} seconds...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    recording = False
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file_name = temp_audio_file.name
        sf.write(temp_audio_file_name, audio_data, fs)
        return temp_audio_file_name

def process_audio(temp_audio_file_name):
    with st.spinner("Processing audio..."):
        try:
            user_input = transcribe_audio(temp_audio_file_name)
            if user_input and "Error" not in user_input:
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
                    # Audio Synthesis
                    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    synthesize_speech(ai_response, temp_audio_file.name)
                    play_audio(temp_audio_file.name)
                    os.remove(temp_audio_file_name) #remove temp audio file after processing.
                    st.rerun()
            else:
                st.error(user_input)
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logging.exception("An unexpected error occurred during audio processing")


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
    # Audio Synthesis
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    synthesize_speech(ai_response, temp_audio_file.name)
    play_audio(temp_audio_file.name)
    st.rerun()


# --- Audio Input/Processing Button (separate from chat input)---
if "sounddevice" in sys.modules:
    if st.button("Record and Process Audio", key="audio_button"):
        if not recording:
            st.session_state['audio_button_clicked'] = True
            temp_audio_file_name = record_audio()
            process_audio(temp_audio_file_name)