import os
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
    file_path = os.path.abspath("./intents.json")
    with open(file_path, "r") as file:
        intents = json.load(file)
    return ChatBot(intents)

def main():
    st.title("Intents of Chatbot using NLP")
    
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
    st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on intents...")
    # Rest of the about section content...

if __name__ == '__main__':
    main() 
