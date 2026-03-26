from src.helper import Huggingface_embedding_model
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from src.prompt import system_prompt
import os

# Load environment variables
load_dotenv()

# Get API keys from .env
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Validate keys
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing in .env file")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing in .env file")

# Set Pinecone API key
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Load embedding model
embeddings = Huggingface_embedding_model()

# Pinecone index name
INDEX_NAME = "medicalbot"

# Connect to existing Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

# Retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.6,
    max_tokens=500
)

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# Create RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(page_title="Medi Bot", page_icon="🩺")
st.title("🩺 Medi Bot")
st.subheader("How can I help you? 👨‍⚕️")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_query = st.chat_input("Type your medical question here...")

if user_query:
    # Show user message
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Generate response
    with st.spinner("Please wait..."):
        try:
            response = rag_chain.invoke({"input": user_query})
            answer = response.get("answer", "Sorry, I couldn't find an answer.")
        except Exception as e:
            answer = f"Error: {str(e)}"

    # Show assistant response
    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})