import streamlit as st
import os
import time
from PIL import Image
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# 🖼️ Display banner image with custom width and height
banner = Image.open("genai_images.jpg")  # Updated filename here
banner_resized = banner.resize((1200, 250))  # Adjust dimensions as needed
st.image(banner_resized)

st.title("📚 Chat with Your PDF Documents")

# Load API keys from environment
os.environ['GROQ_API_KEY'] = st.secrets.get("GROQ_API_KEY", "")
#os.environ['OPENAI_API_KEY'] = st.secrets.get("OPENAI_API_KEY", "")
#os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY', '')
#os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', '')

if not os.environ['GROQ_API_KEY'] :
    st.error("🚨 Please set your GROQ_API_KEY and OPENAI_API_KEY environment variables.")
    st.stop()

# Initialize LLM
llm = ChatGroq(model_name="Llama3-8b-8192")

# Prompt template (must include 'context')
prompt = ChatPromptTemplate.from_template(
    """
    You are an expert assistant. Use the following context to answer the question accurately.
    Do not include phrases like "The context says..." or "Based on the context...".
    If the answer is not in the context, say "I don't know."

    Context:
    {context}

    Question: {input}
    """
)

# Vector database loading & caching
@st.cache_resource(show_spinner="🔄 Creating vector store from documents...")
def load_embeddings():
    embeddings = OllamaEmbeddings()
    loader = PyPDFDirectoryLoader("documents")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_split_doc = text_splitter.split_documents(docs)
    return FAISS.from_documents(final_split_doc, embeddings)

# Automatically load vector store
vector_store = load_embeddings()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_prompt = st.text_input("💬 Ask a question about your PDFs")

# Q&A logic
if user_prompt:
    # Save user prompt to chat history
    st.session_state.chat_history.append(f"**You:** {user_prompt}")

    doc_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt})
    duration = time.process_time() - start

    # Save AI response to chat history
    ai_response = response.get("answer", "No answer found.")
    st.session_state.chat_history.append(f"**AI:** {ai_response}")

    # Display chat history
    for message in st.session_state.chat_history:
        st.write(message)
