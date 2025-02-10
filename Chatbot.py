import os
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Cache this function so we don't re-read and re-index PDFs every time we refresh
@st.cache_resource(show_spinner=False)
def index_pdf_files(pdf_dir):
    """
    Reads all PDF files from the given directory, splits them into smaller chunks,
    and indexes them for quick search later on.
    """
    # We'll gather all the content from each PDF in a list
    all_pdf_content = []

    # Loop through each file in the folder
    for file_name in os.listdir(pdf_dir):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(pdf_dir, file_name)
            # Load the PDF using our handy loader
            pdf_loader = PyPDFLoader(file_path)
            # Add the document(s) from this PDF into our list
            all_pdf_content.extend(pdf_loader.load())

    # Split the text into chunks to make it easier for the model to process.
    # This also helps with handling long documents.
    text_chunker = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_texts = text_chunker.split_documents(all_pdf_content)

    # Generate embeddings using our chosen model.
    embedding_model = OllamaEmbeddings(model="llama3")
    # Build an index (using FAISS) so we can quickly find relevant chunks later.
    pdf_index = FAISS.from_documents(split_texts, embedding_model)
    return pdf_index

# Cache this function so our QA chain is set up just once.
@st.cache_resource(show_spinner=False)
def setup_qa_bot(indexed_pdfs):
    """
    Sets up the question-answering chain using our PDF index and a chat model.
    This chain will fetch relevant document chunks and generate answer.
    """
    # Get a retriever from our index that will find relevant text
    doc_retriever = indexed_pdfs.as_retriever()
    # Initialize our chat model 
    chat_model = ChatOllama(model="llama3")
    # Build the QA chain that ties everything together
    qa_bot = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",  # We're just stuffing the text chunks together
        retriever=doc_retriever
    )
    return qa_bot

def get_voice_query():
    """
    Listens to the user's voice using the microphone and converts it to text.
    Returns the recognized text or None if something goes wrong.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as mic:
        st.info("🎙 Please speak your question now...")
        try:
            # Listen for about 5 seconds for the user's speech
            audio_input = recognizer.listen(mic, timeout=5)
            # Convert the audio into text using Google's API
            spoken_query = recognizer.recognize_google(audio_input)
            st.success(f"🗣 You said: {spoken_query}")
            return spoken_query
        except sr.UnknownValueError:
            st.error("😕 Sorry, I couldn't understand your speech.")
        except sr.RequestError:
            st.error("❌ Oops, there was a problem with the speech recognition service.")
    return None

def text_to_audio(answer_text):
    """
    Converts a text string into speech using gTTS, saves it as a temporary MP3 file,
    and returns the file path.
    """
    tts = gTTS(text=answer_text, lang="en")
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name

def main():
    st.title("🗣 PDF Voice Chatbot")
    st.write("Ask any questions about the PDFs in the **Data/** folder using your keyboard or your voice.")

    # Folder where our PDFs are stored
    pdf_folder_path = "Data/"

    # Load and index the PDFs, then set up our chatbot
    with st.spinner("🔄 Processing PDFs..."):
        pdf_index = index_pdf_files(pdf_folder_path)
        qa_bot = setup_qa_bot(pdf_index)

    st.success("✅ PDFs are loaded and ready to go!")

    # Let the user type a question
    user_question = st.text_input("🔍 Type your question here:")

    # If they prefer speaking, they can click the button
    if st.button("🎙 Speak"):
        user_question = get_voice_query()

    # If we got a question from either input method, get an answer
    if user_question:
        with st.spinner("🤖 Thinking..."):
            bot_answer = qa_bot.run(user_question)
        
        st.markdown("### 📝 Chatbot's Answer:")
        st.write(bot_answer)

        # Convert the text answer to audio and play it
        audio_path = text_to_audio(bot_answer)
        st.audio(audio_path, format="audio/mp3")

if __name__ == "__main__":
    main()
