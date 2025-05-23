import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# Pinecone vector DB and specification
from pinecone import Pinecone, ServerlessSpec

# LlamaIndex core components
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
# from llama_index.vector_stores import PineconeVectorStore
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.readers.file import PyMuPDFReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.memory import ChatMemoryBuffer

# LangChain utility for text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()


# Configuration constants
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
PDF_FOLDER = Path(r"D:\Ahsan\Office\Data")  # Path to PDF files
PINECONE_INDEX_NAME = "myself"  # Name of the Pinecone index
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PROCESSED_TRACKER = "processed_files.json"  # File to track already processed PDFs


# System prompt guiding the chatbot's tone and behavior

SYSTEM_PROMPT = """\
You are a helpful, knowledgeable, and friendly AI assistant. 
When answering a question:
1. First use ONLY the context from the provided documents.
2. If the answer is not found in the context, say "I don't know based on the provided documents."
3. Then, you may add relevant information from your general knowledge if available.

Format your response as:
[Document-based answer]
[Additional Information from AI (if applicable)]
"""

# Load previously processed PDF hashes (to avoid reprocessing)
def load_processed_hashes():
    if os.path.exists(PROCESSED_TRACKER):
        with open(PROCESSED_TRACKER, 'r') as f:
            return set(json.load(f))
    return set()

# Save updated list of processed file hashes
def save_processed_hashes(hashes):
    with open(PROCESSED_TRACKER, 'w') as f:
        json.dump(list(hashes), f)

# Generate a unique hash for a PDF file's content
def hash_file(path):
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()


# Load new PDFs (not previously processed), return documents and their hashes
def load_new_documents(folder, processed_hashes):
    reader = PyMuPDFReader()
    documents = []
    new_hashes = set()

    for pdf in folder.glob("*.pdf"):
        file_hash = hash_file(pdf)
        if file_hash not in processed_hashes:
            documents.extend(reader.load_data(file_path=str(pdf)))
            new_hashes.add(file_hash)

    return documents, new_hashes


# Split PDF documents into smaller text chunks
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for doc in documents:
        for chunk in splitter.split_text(doc.text):
            chunks.append(Document(text=chunk, metadata=doc.metadata))
    return chunks


# Initialize and connect to Pinecone vector database
def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create the index only if it doesn't exist
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,  # Must match embedding output size
            metric='dotproduct',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    return pc.Index(PINECONE_INDEX_NAME)




# ---------- MAIN EXECUTION FLOW ----------


if __name__ == "__main__":
    # Check required environment variables
    print("Checking API Keys...")
    assert PINECONE_API_KEY and GROQ_API_KEY and PINECONE_ENV, "Missing environment variables."

    # Step 1: Connect to Pinecone
    print("Initializing Pinecone...")
    pinecone_index = init_pinecone()

    # Step 2: Wrap the Pinecone index in LlamaIndex vector store interface
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Step 3: Load already processed file hashes
    print("Loading previously processed files...")
    processed_hashes = load_processed_hashes()

    # Step 4: Load and hash new documents
    print("Loading new documents...")
    raw_docs, new_hashes = load_new_documents(PDF_FOLDER, processed_hashes)

    # Step 5: Embed new documents and add to index
    if raw_docs:
        print("Splitting and embedding new documents...")
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
        Settings.embed_model = embed_model

        # Split and index chunks
        chunks = split_documents(raw_docs)
        VectorStoreIndex.from_documents(chunks, storage_context=storage_context)

        # Track newly processed files
        processed_hashes.update(new_hashes)
        save_processed_hashes(processed_hashes)
    else:
        # No new data: connect to existing index
        print("No new documents found. Connecting to existing vector store...")
        # additionally, we need to set the embedding model again as it will use openai's default otherwise
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
        Settings.embed_model = embed_model
    
    # Step 6: Reconnect index (existing or newly built)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store,storage_context=storage_context)

    # Step 7: Configure and initialize Groq LLM
    print("Initializing LLM...")
    Settings.llm = Groq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        api_key=GROQ_API_KEY,
        max_tokens=700,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        num_beams=1,
        do_sample=False,
        
        
        
    )


    # Step 8: Create chat engine with short-term conversational memory
    print("Starting chat engine...")
    memory = ChatMemoryBuffer.from_defaults(token_limit=2000)
    chat_engine = index.as_chat_engine(system_prompt=SYSTEM_PROMPT, chat_mode="context", memory=memory)
    
    

    # Step 9: User input loop
    print("Chatbot is ready! Type 'exit' || 'quit' || 'close' || 'bye' to end the conversation.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit", "close", "bye"}:
            print("Goodbye!")
            break
        try:
            # Query chatbot with user's input
            response = chat_engine.chat(user_input)
            print(f"Chatbot: {response}")
        except Exception as e:
            print(f"Error: {str(e)}")

                


