import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import json
import nltk
import ssl
import streamlit as st
from chatbot import ChatBot
from chat_history import ChatHistory

# SSL Configuration and NLTK Downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Create nltk_data directory if it doesn't exist
nltk_data_dir = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

# Initialize components
@st.cache_resource
def initialize_chatbot():
    file_path = os.path.join(os.path.dirname(__file__), "intents.json")
    st.write(f"Looking for intents.json at: {file_path}")  # Debug line
    try:
        with open(file_path, "r", encoding='utf-8') as file:
            intents = json.load(file)
            if not isinstance(intents, dict) or "intents" not in intents:
                raise ValueError("Invalid intents.json format")
            return ChatBot(intents)
    except FileNotFoundError:
        st.error(f"Could not find intents.json at {file_path}")
        raise
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON in intents.json: {str(e)}")
        raise
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        raise

def main():
    st.title(" Chatbot using NLP")
    
    chatbot = initialize_chatbot()
    chat_history = ChatHistory()
    
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")
        
        # Get user input
        user_input = st.text_input("You:", key="user_input")
        
        if user_input:
            # Get chatbot response
            response = chatbot.get_response(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key="chatbot_response")
            
            # Save conversation
            chat_history.add_conversation(user_input, response)
            
            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.header("Conversation History")
        for conv in chat_history.get_history():
            st.text(f"User: {conv['user_input']}")
            st.text(f"Chatbot: {conv['response']}")
            st.text(f"Timestamp: {conv['timestamp']}")
            st.markdown("---")

    elif choice == "About":
        display_about_section()

def display_about_section():
    st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using Natural Language Processing (NLP) library and Logistic Regression.")

    st.subheader("Project Overview:")
    st.write("""
    The project is divided into two parts:
    1. NLP techniques and Logistic Regression algorithm is used to train the chatbot on labeled intents and entities.
    2. For building the Chatbot interface, Streamlit web framework is used to build a web-based chatbot interface.
    """)

    st.subheader("Features:")
    st.write("""
    - Natural language understanding using NLTK
    - Machine learning-based intent classification
    - Conversation history tracking
    - Simple and intuitive web interface
    - Real-time responses
    """)

if __name__ == '__main__':
    main() 
