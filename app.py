
from src.helper import Huggingface_embedding_model
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from src.prompt import *
import os


load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY


embeddings = Huggingface_embedding_model()

index_name = "medicalbot"

# embedded each chunk and upsert the embeddings

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})



    
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-04-17",
    google_api_key = "AIzaSyCYaIDh7ODn0n8F88JbVhRjIYqSo_ZJTbk",
    temperature=0.6,
    max_tokens=500
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human","{input}")
    ]
    
)
question_answer_chain = create_stuff_documents_chain(llm,prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# --- UI ---
st.title("🩺 Medi Bot")
st.subheader("How Can i help you 🧑‍⚕️")

# Display chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_query = st.chat_input("Type your question...")

if user_query:
    # Show user message
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Get response
    with st.spinner("Please Wait.."):
        response = rag_chain.invoke({"input": user_query})
        answer = response.get("answer", "Sorry, I couldn't find an answer.")

    # Show assistant response
    st.chat_message("assistant").markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})