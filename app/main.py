import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from app.components.chat_interface import display_chat_messages, get_user_input
from app.utils.vector_db import load_and_split_documents, initialize_vector_db, perform_similarity_search
from app.utils.gemini_utils import generate_response, build_rag_prompt
from app.audio.audio_processing import transcribe_audio, synthesize_speech, play_audio
import tempfile
import streamlit as st
import time
from gtts import gTTS
import logging
import uuid
import shutil
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s - %(lineno)d - %(message)s')

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

# --- Audio Input/Processing ---
if 'recording' not in st.session_state:
    st.session_state.recording = False
audio_data = None

audio_dir = "audio_files"
os.makedirs(audio_dir, exist_ok=True)

def cleanup_old_audio_files(audio_dir, cutoff_time):
    for filename in os.listdir(audio_dir):
        file_path = os.path.join(audio_dir, filename)
        file_creation_time = os.path.getmtime(file_path)
        if time.time() - file_creation_time > cutoff_time:
            try:
                os.remove(file_path)
                logging.info(f"Deleted old audio file: {file_path}")
            except OSError as e:
                logging.error(f"Error deleting audio file {file_path}: {e}")

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
    file_id = str(uuid.uuid4())
    temp_audio_file_name = os.path.join(audio_dir, f"{file_id}.wav")
    sf.write(temp_audio_file_name, audio_data, fs)
    logging.info(f"Audio recording saved to: {temp_audio_file_name}")
    return temp_audio_file_name

def process_audio(temp_audio_file_name):
    with st.spinner("Processing audio..."):
        try:
            user_input = transcribe_audio(temp_audio_file_name)
            if user_input and "Error" not in user_input:
                st.session_state.chat_history.append(("user", user_input))
                with st.chat_message("user"):
                        st.write(user_input)
                with st.spinner("Fetching answer using RAG technique ..."):
                    # Perform Similarity Search
                    relevant_documents = perform_similarity_search(st.session_state.vector_db, user_input)
                    if relevant_documents:
                        context = "\n\n".join(relevant_documents)
                        rag_prompt = build_rag_prompt(user_input, context)
                        ai_response = generate_response(rag_prompt)
                    else:
                        ai_response = generate_response(user_input)
                    with st.chat_message("ai"):
                        st.write(ai_response)
                    # Audio Synthesis
                    file_id = str(uuid.uuid4())
                    audio_response_file = os.path.join(audio_dir, f"{file_id}.mp3")
                    synthesize_speech(ai_response, audio_response_file)
                    #Play audio response
                    logging.info(f"Audio response generated at: {audio_response_file}")
                    audio_file = open(audio_response_file, "rb")
                    audio_bytes = audio_file.read()
                    with st.chat_message("ai"):
                        st.audio(audio_bytes, format='audio/mpeg')
                    audio_file.close()
                    # Remove temporary files after playing audio.
                    os.remove(temp_audio_file_name)
                    st.session_state.chat_history.append(("ai", ai_response))
                    
            else:
                st.error(user_input)
                logging.error(f"Transcription failed: {user_input}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logging.exception("An unexpected error occurred during audio processing")

# --- Chat Interface ---
display_chat_messages(st.session_state.chat_history)

user_input = get_user_input()

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
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
    with st.chat_message("ai"):
        st.write(ai_response)
    st.session_state.chat_history.append(("ai", ai_response))
    # Audio Synthesis
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    synthesize_speech(ai_response, temp_audio_file.name)
    play_audio(temp_audio_file.name)
    

# --- Audio Input/Processing Button (separate from chat input)---
try:
    import sounddevice as sd
    import soundfile as sf
    if st.button("Record and Ask", key="audio_button"):
        if not st.session_state.recording:
            temp_audio_file_name = record_audio()
            process_audio(temp_audio_file_name)
except ImportError:
    st.info("Audio recording is not supported in this environment.")

cleanup_old_audio_files(audio_dir, 3600) #cleanup files older than 1 hour.