import streamlit as st
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import random

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Initialize NLTK components
lemmatizer = nltk.WordNetLemmatizer()

# Load intents
intents = {
    "intents": [
        {
            "tag": "greeting",
            "patterns": ["Hi", "Hello", "Hey", "How are you", "Good day"],
            "responses": ["Hello!", "Hi there!", "Hey! How can I help you?"]
        },
        {
            "tag": "goodbye",
            "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
            "responses": ["Goodbye!", "See you later!", "Take care!"]
        },
        {
            "tag": "thanks",
            "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
            "responses": ["You're welcome!", "Happy to help!", "My pleasure!"]
        }
        # Add more intents as needed
    ]
}

def preprocess_text(text):
    # Tokenize and lemmatize
    tokens = nltk.word_tokenize(text.lower())
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])

def get_response(user_input):
    # Preprocess user input
    processed_input = preprocess_text(user_input)
    
    # Prepare all patterns for comparison
    all_patterns = []
    for intent in intents["intents"]:
        all_patterns.extend(intent["patterns"])
    
    # Add user input to patterns for vectorization
    all_patterns.append(processed_input)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_patterns)
    
    # Calculate similarity between user input and all patterns
    user_vector = tfidf_matrix[-1]
    pattern_vectors = tfidf_matrix[:-1]
    similarities = cosine_similarity(user_vector, pattern_vectors)
    
    # Find the most similar pattern
    max_similarity_index = np.argmax(similarities)
    max_similarity = similarities[0][max_similarity_index]
    
    # If similarity is too low, return default response
    if max_similarity < 0.3:
        return "I'm not sure how to respond to that. Could you please rephrase?"
    
    # Find the intent that contains the most similar pattern
    pattern_counter = 0
    for intent in intents["intents"]:
        if pattern_counter <= max_similarity_index < pattern_counter + len(intent["patterns"]):
            return random.choice(intent["responses"])
        pattern_counter += len(intent["patterns"])
    
    return "I'm not sure how to respond to that."

# Streamlit UI
st.title("AI Chatbot")
st.write("Hello! How can I help you today?")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What's on your mind?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    response = get_response(prompt)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# Sidebar with information
with st.sidebar:
    st.title("About")
    st.write("This is a simple chatbot built with Streamlit, NLTK, and scikit-learn.")
    st.write("It uses TF-IDF vectorization and cosine similarity to find the most appropriate response.") 