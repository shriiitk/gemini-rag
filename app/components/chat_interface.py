import streamlit as st
from typing import List, Tuple

def display_chat_messages(chat_history: List[Tuple[str, str]]) -> None:
    """
    Displays chat messages in the chat interface.

    Args:
        chat_history (List[Tuple[str, str]]): A list of tuples, where each tuple
                                               contains the speaker ('user' or 'ai')
                                               and the message.
    """
    for speaker, text in chat_history:
        if speaker == 'user':
            with st.chat_message("user"):
                st.write(text)
        elif speaker == 'ai':
            with st.chat_message("ai"):
                st.write(text)


def get_user_input() -> str:
    """
    Gets the user's input from the chat input.

    Returns:
        str: The user's input text.
    """
    user_input = st.chat_input("Type your question here...")
    return user_input if user_input else ""