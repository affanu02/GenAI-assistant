import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

# Load openAI key into OPENAI_API_KEY from personal secure .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# OPENAI_API_KEY = "key-goes-here, if you want to copy straight from your OpenAI Account"

# Name of AI chatbot
st.header("Affans AI Assistant")

# Upload PDF files
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a pdf file and start asking questions.", type="pdf")

# Extract text from documents
if file is not None:
    pdf_reader = PdfReader(file)
    text=""
    
    for page in pdf_reader.pages:
        text += page.extract_text()
        #st.write(text)

    # Break mass text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=2000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks)


    # Generating embeddings
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)

    # Creating the vector store - FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Get user questions
    user_question = st.text_input("Type your question here:")

    # Do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        # st.write(match)

        # Define LLM
        llm = ChatOpenAI(
            openai_api_key = OPENAI_API_KEY,
            temperature = 0,
            max_tokens = 2000,
            model_name = "gpt-3.5-turbo"
        )

        # Output results
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents = match, question = user_question)
        st.write(response)
