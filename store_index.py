from src.helper import load_pdf_file, text_split, Huggingface_embedding_model
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing in .env file")

# Load and process PDFs
print("📄 Loading PDF files...")
extracted_data = load_pdf_file(data="data/")
print(f"✅ Loaded {len(extracted_data)} documents.")

print("✂️ Splitting text into chunks...")
text_chunks = text_split(extracted_data)
print(f"✅ Created {len(text_chunks)} text chunks.")

print("🧠 Loading embedding model...")
embeddings = Huggingface_embedding_model()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "medicalbot"

# Check if index exists
existing_indexes = pc.list_indexes().names()

if INDEX_NAME not in existing_indexes:
    print(f"📦 Creating Pinecone index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("⏳ Waiting for index to be ready...")
    time.sleep(10)
else:
    print(f"✅ Index '{INDEX_NAME}' already exists.")

# Upload documents
print("⬆️ Uploading embeddings to Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=INDEX_NAME,
    embedding=embeddings
)

print("✅ All documents uploaded successfully to Pinecone!")