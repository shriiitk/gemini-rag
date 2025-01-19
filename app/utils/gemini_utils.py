import google.generativeai as genai
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=GOOGLE_API_KEY)

def generate_response(prompt: str) -> str:
    """
    Generates a response from the Gemini Pro model based on the given prompt.

    Args:
        prompt (str): The prompt text to send to the model.

    Returns:
        str: The generated response text.
    """
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return "Sorry, I encountered an error processing your request."


def build_rag_prompt(query: str, context: str) -> str:
    """
    Builds a RAG prompt by combining the user query and the relevant context.

    Args:
        query (str): The user's query string.
        context (str): The relevant context extracted from vector DB

    Returns:
        str: The combined prompt string.
    """
    prompt = f"""
    Given the following context:
    {context}

    Answer the following question:
    {query}
    
    If the answer is not available in context say that you don't know.
    """
    return prompt