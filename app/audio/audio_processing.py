import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
import os
import streamlit as st
import assemblyai as aai

def transcribe_audio(audio_file):
    """Transcribes audio using AssemblyAI."""
    aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
    transcriber = aai.Transcriber()
    try:
        transcript = transcriber.transcribe(audio_file)
        if transcript.status == aai.TranscriptStatus.error:
            return f"Transcription failed: {transcript.error}"
        return transcript.text
    except Exception as e:
        return f"Error during AssemblyAI transcription: {e}"

def synthesize_speech(text, filename):
    """Synthesizes text to speech using gTTS."""
    tts = gTTS(text=text, lang='en')
    tts.save(filename)

def play_audio(filename):
    """Plays audio file using streamlit's audio component."""
    try:
        audio_file = open(filename, "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes) #removed type argument
        audio_file.close()
        os.remove(filename)  # Clean up after playing
    except FileNotFoundError:
        st.error("Audio file not found")