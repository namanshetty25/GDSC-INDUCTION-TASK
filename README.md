PDF Voice Chatbot

Overview
The PDF Voice Chatbot(MetaMind) is a Streamlit-based application that enables users to interact with PDFs using text or voice queries. It processes PDFs, extracts their content, and allows users to ask questions about the documents using an AI-powered chatbot.


Features

PDF Processing: Loads and indexes PDFs from the Data/ folder.

AI-powered Q&A: Uses LangChain and Ollama to retrieve relevant document chunks and generate answers.

Voice Input Support: Converts speech to text using SpeechRecognition.

Text-to-Speech (TTS): Converts chatbot responses into speech using gTTS.

Fast Retrieval: Uses FAISS for efficient document search.


Installation

Ensure you have Python installed, then install the required dependencies:

pip install -r requirements.txt


Project Structure

📁 PDF-Voice-Chatbot

│── 📁 Data/               # Folder containing PDFs to be processed

│── 📄 MetaMind.py              # Main Streamlit app

│── 📄 README.md           # Project documentation

│── 📄 requirements.txt    # List of dependencies


Technologies Used

Streamlit - For building the user interface.

LangChain - For document processing and AI chatbot integration.

FAISS - For fast and efficient document retrieval.

Ollama - For language model-based answers.

SpeechRecognition - For converting speech to text.

gTTS - For converting text responses to speech.


Feel free to contribute by submitting pull requests or reporting issues. Let's improve this chatbot together.

License

This project is licensed under the MIT License.
