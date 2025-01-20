import os
import streamlit as st
import assemblyai as aai
from gtts import gTTS
import logging

def transcribe_audio(audio_file):
    """Transcribes audio using AssemblyAI."""
    logging.info(f"Starting audio transcription for: {audio_file}")
    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
    transcriber = aai.Transcriber()
    try:
        transcript = transcriber.transcribe(audio_file)
        if transcript.status == aai.TranscriptStatus.error:
            logging.error(f"Transcription failed: {transcript.error}")
            return f"Transcription failed: {transcript.error}"
        logging.info(f"Transcription completed successfully.")
        return transcript.text
    except Exception as e:
        logging.exception(f"Error during AssemblyAI transcription: {e}")
        return f"Error during AssemblyAI transcription: {e}"

def synthesize_speech(text, filename):
    """Synthesizes text to speech using gTTS."""
    logging.info(f"Starting text-to-speech synthesis for: {text}")
    tts = gTTS(text=text, lang='en')
    try:
        tts.save(filename)
        logging.info(f"Audio file saved to: {filename}")
    except Exception as e:
        logging.exception(f"Error during text-to-speech synthesis: {e}")
        return f"Error during text-to-speech synthesis: {e}"

def play_audio(filename):
    """Plays audio file using streamlit's audio component."""
    try:
        logging.info(f"Starting audio playback for: {filename}")
        audio_file = open(filename, "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mpeg')
        audio_file.close()
        logging.info(f"Audio playback completed successfully.")
    except FileNotFoundError:
        logging.error(f"Audio file not found: {filename}")
        st.error("Audio file not found")
    except Exception as e:
        logging.exception(f"Error during audio playback: {e}")
        st.error("Error during audio playback")