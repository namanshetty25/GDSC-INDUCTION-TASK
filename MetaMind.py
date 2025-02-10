import os
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import tempfile

# These are our LangChain helpers for loading PDFs, embedding text, and performing QA.
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# This function loads all PDFs from a given folder, splits their content into chunks,
# and then creates a searchable index (vectorstore) from the text.
@st.cache_resource(show_spinner=False)
def load_and_index_pdfs(pdf_folder):
    # Create an empty list to store all document content
    documents = []
    
    # Loop over each file in the specified folder
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            # For every PDF, we use the PyPDFLoader to read its content
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents.extend(loader.load())
    
    # We use a text splitter to break long documents into smaller chunks.
    # This is useful for more precise and efficient querying later.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    
    # Now we convert the text chunks into embeddings using the OllamaEmbeddings model.
    # Embeddings transform text into numerical representations that make similarity searches possible.
    embeddings = OllamaEmbeddings(model="llama3")
    
    # We then build a FAISS index, which is a vector store that can quickly find similar text chunks.
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    
    # Return the searchable vector index so we can later retrieve relevant parts of the PDFs.
    return vectorstore

# This function sets up our QA chain. It creates a retriever from the vectorstore
# and then configures a chatbot model that will generate answers.
@st.cache_resource(show_spinner=False)
def load_qa_chain(_vectorstore):
    # Turn the vectorstore into a retriever that can find the most relevant document chunks.
    retriever = _vectorstore.as_retriever()
    
    # Initialize the chatbot model. Here we're using ChatOllama with the "llama3" model.
    chat_model = ChatOllama(model="llama3")
    
    # Create a RetrievalQA chain that ties the retriever and chat model together.
    # This chain will find relevant text chunks and then use the LLM to produce an answer.
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model,
        chain_type="stuff",  # 'stuff' indicates a simple concatenation of retrieved chunks
        retriever=retriever
    )
    
    return qa_chain

# This function listens to microphone and uses speech recognition to convert your speech into text.
def recognize_speech():
    # Initialize the speech recognizer
    recognizer = sr.Recognizer()
    
    # Open the microphone and listen for a few seconds.
    with sr.Microphone() as source:
        st.info("🎙 Speak your query now...")  # Inform the user to start talking
        try:
            # Listen for the user's speech (with a timeout of 5 seconds)
            audio = recognizer.listen(source, timeout=5)
            # Use Google's speech recognition to transcribe the audio to text
            query = recognizer.recognize_google(audio)
            st.success(f"🗣 You said: {query}")
            return query
        except sr.UnknownValueError:
            # This error happens if the speech is not clear or understandable
            st.error("😕 Could not understand the audio.")
        except sr.RequestError:
            # This error happens if there's a problem with the speech recognition service
            st.error("❌ Speech recognition service error.")
    
    # If something went wrong, return None
    return None

# This function takes text and converts it into speech (an mp3 file) using gTTS.
def speak_text(text):
    # Create a gTTS object with the text provided and language set to English.
    tts = gTTS(text=text, lang="en")
    # Create a temporary file to save the audio
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

# This is the main function that sets up our Streamlit app.
def main():
    st.title("🗣 PDF Voice Chatbot")
    st.write("Ask questions about the PDFs loaded from the **Data/** folder using **text or voice**.")

    # Define the folder where PDFs are stored. You can change this if needed.
    pdf_folder = "Data/"

    # Inform the user that we're processing the PDFs.
    with st.spinner("🔄 Loading and indexing PDFs..."):
        # Load PDFs and create a searchable vectorstore
        vectorstore = load_and_index_pdfs(pdf_folder)
        # Set up the QA chain with our chatbot model
        qa_chain = load_qa_chain(vectorstore)

    st.success("✅ PDFs loaded and indexed successfully!")

    # Allow the user to enter a question via a text input box
    query = st.text_input("🔍 Enter your question:")
    
    # If the user clicks the "Speak" button, we use voice recognition instead
    if st.button("🎙 Speak"):
        query = recognize_speech()
        
    # If there's a valid query (either text or voice), generate an answer
    if query:
        with st.spinner("🤖 Generating answer..."):
            response = qa_chain.run(query)
        
        # Display the chatbot's response on the page
        st.markdown("### 📝 Chatbot Response:")
        st.write(response)

        # Convert the response into speech so the user can listen to it
        speech_file = speak_text(response)
        st.audio(speech_file, format="audio/mp3")

if __name__ == "__main__":
    main()
