import streamlit as st
import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

# Pinecone vector DB and specification
from pinecone import Pinecone, ServerlessSpec

# LlamaIndex core components
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
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
PDF_FOLDER = Path(os.getenv('DATA_PATH') ) # Path to PDF files
PINECONE_INDEX_NAME = "chat-pdf"  # Name of the Pinecone index
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PROCESSED_TRACKER = "processed_files.json"  # File to track processed PDFs

SYSTEM_PROMPT = """\
You are a helpful, knowledgeable, and friendly AI assistant.
When answering a question:
1. Use ONLY the context from the provided documents to answer.
2. For every answer, clearly cite the source document(s) and, if available, provide a direct link or reference to the relevant section or page.
3. If the answer is not found in the context, say "I don't know based on the provided documents."
4. After your document-based answer, you may add relevant information from your general knowledge if available, and clearly separate it from the document-based answer.

Format your response as:
[Document-based answer, with in-text citations or links to the source document(s)]
[Additional Information from AI (if applicable)]
[Sources: <source document name(s)> and direct link(s) or citation(s) to the referenced data]
"""



# Load processed files data (hashes and filenames)
def load_processed_files():
    """Load previously processed file hashes and filenames from JSON file"""
    if os.path.exists(PROCESSED_TRACKER):
        try:
            with open(PROCESSED_TRACKER, 'r') as f:
                data = json.load(f)
                # Check if data is a dictionary and has the expected keys
                if isinstance(data, dict) and "hashes" in data and "filenames" in data:
                    return data
                else:
                    # If data is not in the expected format, return empty data
                    st.warning("Processed files data is corrupted. Starting with fresh data.")
                    print("Processed files data is corrupted. Starting with fresh data.")
                    return {"hashes": [], "filenames": []}
        except json.JSONDecodeError:
            # If the JSON is invalid, return empty data
            st.warning("Processed files JSON is invalid. Starting with fresh data.")
            print("Processed files JSON is invalid. Starting with fresh data.")
            return {"hashes": [], "filenames": []}
        except Exception as e:
            # Catch any other exceptions and return empty data
            st.warning(f"Error loading processed files: {str(e)}. Starting with fresh data.")
            print(f"Error loading processed files: {str(e)}. Starting with fresh data.")
            return {"hashes": [], "filenames": []}
    return {"hashes": [], "filenames": []}




# Save updated list of processed files
def save_processed_files(hashes, filenames):
    """Save processed file hashes and filenames to JSON file"""
    data = {
        "hashes": list(hashes),
        "filenames": filenames
    }
    with open(PROCESSED_TRACKER, 'w') as f:
        json.dump(data, f)

# Generate a unique hash for a PDF file's content
def hash_file(path):
    """Generate MD5 hash for file content"""
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()

# Load new PDFs (not previously processed), return documents and their hashes
def load_new_documents(folder, processed_hashes):
    """Load new PDF documents from folder that haven't been processed before"""
    reader = PyMuPDFReader()
    documents = []
    new_hashes = set()
    new_filenames = []

    for pdf in folder.glob("*.pdf"):
        file_hash = hash_file(pdf)
        if file_hash not in processed_hashes:
            documents.extend(reader.load_data(file_path=str(pdf)))
            new_hashes.add(file_hash)
            new_filenames.append(pdf.name)
    
    return documents, new_hashes, new_filenames



# Split PDF documents into smaller text chunks
def split_documents(documents):
    """Split documents into smaller chunks for better vector storage"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for doc in documents:
        for chunk in splitter.split_text(doc.text):
            chunks.append(Document(text=chunk, metadata=doc.metadata))
    return chunks

# Initialize and connect to Pinecone vector database
def init_pinecone():
    """Initialize connection to Pinecone vector database"""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create index if it doesn't exist
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric='dotproduct',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    return pc.Index(PINECONE_INDEX_NAME)




# Streamlit UI with modifications
def main():
    """Main function to initialize Streamlit UI"""
    st.set_page_config(page_title="PDF ChatBot", page_icon=":books:", layout="wide")
    
    # Custom CSS for styling
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            width: 220px !important; /* Reduced sidebar width */
        }
        
        .sidebar-content {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        
        .profile-image {
            width: 100px;
            height: 100px;
            border-radius: 50%; /* Circular image */
            margin-bottom: 20px;
        }
        
        .message {
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 8px;
        }
        
        .user-message {
            background-color: #2d2d2d;
        }
        
        .bot-message {
            background-color: #333333;
        }
        
        .document-list {
            max-height: 300px;
            overflow-y: auto;
            margin-top: 20px;
        }
        
        .file-item {
            display: flex;
            justify-content: space-between;
            padding: 8px;
            border-bottom: 1px solid #444;
        }
        
        .chat-history {
            margin-top: 20px;
            padding: 10px;
            background-color: #1a1a1a;
            border-radius: 8px;
            max-height: 400px;
            overflow-y: auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Sidebar with logo and navigation
    
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.image("https://img.freepik.com/premium-vector/rag-logo_745595-1122.jpg", 
                width=100,
                output_format="PNG",
                caption="MyLogo")  #  image
        navigation = st.radio("Navigate", ["Home", "Chat", "Documents", "Chat History"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    if navigation == "Home":
        show_home_page()
    elif navigation == "Chat":
        show_chat_page()
    elif navigation == "Documents":
        show_documents_page()
    elif navigation == "Chat History":
        show_chat_history()



def show_home_page():
    """Display home page with introduction and features"""
    st.markdown('<h1>PDF ChatBot</h1>', unsafe_allow_html=True)
    
    # Add a project-related image
    st.image("https://miro.medium.com/v2/resize:fit:1100/format:webp/1*Yd5oWBjT7mfmfOh6K-Kt7Q.png", use_column_width=True)
    
    st.markdown('<h2>What is PDF ChatBot?</h2>', unsafe_allow_html=True)
    st.markdown('<p>PDF ChatBot allows you to upload PDF documents and ask questions about their contents. Our AI assistant will use the context from your documents to provide accurate and relevant answers.</p>', unsafe_allow_html=True)
    
    st.markdown('<h2>How to Use</h2>', unsafe_allow_html=True)
    st.markdown('<p>1. Upload your PDF documents in the Documents section</p>', unsafe_allow_html=True)
    st.markdown('<p>2. Navigate to the Chat section to start asking questions</p>', unsafe_allow_html=True)
    st.markdown('<p>3. Refresh documents periodically to load new content</p>', unsafe_allow_html=True)
    
    st.markdown('<h2>Features</h2>', unsafe_allow_html=True)
    st.markdown('<p>- Context-based answering</p>', unsafe_allow_html=True)
    st.markdown('<p>- Document memory and history</p>', unsafe_allow_html=True)
    st.markdown('<p>- Advanced LLM with Groq integration</p>', unsafe_allow_html=True)



# Function to display chat interface
def show_chat_page():
    """Display chat interface with message display"""
    # Initialize session state variables
    if 'chat_engine' not in st.session_state:
        st.session_state.chat_engine = None
        
    if 'message_history' not in st.session_state:
        st.session_state.message_history = []
    
    if 'pinecone_index' not in st.session_state:
        st.session_state.pinecone_index = None
    
    if 'processed_hashes' not in st.session_state:
        st.session_state.processed_hashes = set()
    
    if 'processed_filenames' not in st.session_state:
        st.session_state.processed_filenames = []



    # Initialize Pinecone connection
    if not st.session_state.pinecone_index:
        try:
            st.session_state.pinecone_index = init_pinecone()
            processed_data = load_processed_files()
            st.session_state.processed_hashes = set(processed_data["hashes"])
            st.session_state.processed_filenames = processed_data["filenames"]
        except Exception as e:
            st.error(f"Error initializing Pinecone: {str(e)}")
            return
    
    
    
    # Configure and initialize Groq LLM
    if not st.session_state.chat_engine:
        try:
            embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
            Settings.embed_model = embed_model
            
            vector_store = PineconeVectorStore(pinecone_index=st.session_state.pinecone_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
            
            Settings.llm = Groq(
                model="meta-llama/llama-4-maverick-17b-128e-instruct",
                api_key=GROQ_API_KEY,
                max_tokens=1000,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                num_beams=1,
                do_sample=False,
            )
            
            
            # Create chat engine with short-term conversational memory
            memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
            st.session_state.chat_engine = index.as_chat_engine(system_prompt=SYSTEM_PROMPT, chat_mode="context", memory=memory)
        except Exception as e:
            st.error(f"Error initializing chat engine: {str(e)}")
            return
    
    
    
    # Chat interface
    with st.container():
        st.markdown('<h1>Chat with Your Documents</h1>', unsafe_allow_html=True)
        
        # Chat input form at the bottom
        with st.form(key='chat_form'):
            user_input = st.text_input("You:", key="user_input", value="")
            submitted = st.form_submit_button("Send")
        
            if submitted and user_input and st.session_state.chat_engine:
                try:
                    response = st.session_state.chat_engine.chat(user_input)
                    if not response or not response.response:
                        st.warning("Received empty response from chat engine.")
                    else:
                        st.session_state.message_history.insert(0, ("You", user_input))
                        st.session_state.message_history.insert(1, ("Chatbot\n", response.response))
                    st.session_state.user_input = ""
                except Exception as e:
                    st.error(f"Error processing request: {str(e)}")
        
        # Display chat messages at the top
        for i, (sender, message) in enumerate(reversed(st.session_state.message_history)):
            message_class = "user-message" if sender == "You" else "bot-message"
            st.markdown(
                f'<div class="message {message_class}"><strong>{sender}:</strong> {message}</div>',
                unsafe_allow_html=True
            )


# Function to display document management interface
def show_documents_page():
    """Display document management interface"""
    # Initialize session state variables
    if 'pinecone_index' not in st.session_state:
        st.session_state.pinecone_index = None
    
    if 'processed_hashes' not in st.session_state:
        st.session_state.processed_hashes = set()
    
    if 'processed_filenames' not in st.session_state:
        st.session_state.processed_filenames = []

    # Initialize Pinecone connection
    if not st.session_state.pinecone_index:
        try:
            st.session_state.pinecone_index = init_pinecone()
        except Exception as e:
            st.error(f"Error initializing Pinecone: {str(e)}")
            return
    
    
    
    # Load processed files data
    processed_data = load_processed_files()
    st.session_state.processed_hashes = set(processed_data["hashes"])
    st.session_state.processed_filenames = processed_data["filenames"]
    
    st.markdown('<h1>Document Management</h1>', unsafe_allow_html=True)
    
    # Document management controls
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Refresh Documents"):
            try:
                raw_docs, new_hashes, new_filenames = load_new_documents(PDF_FOLDER, st.session_state.processed_hashes)
                if raw_docs:
                    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
                    Settings.embed_model = embed_model
                    chunks = split_documents(raw_docs)
                    vector_store = PineconeVectorStore(pinecone_index=st.session_state.pinecone_index)
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    VectorStoreIndex.from_documents(chunks, storage_context=storage_context)
                    st.session_state.processed_hashes.update(new_hashes)
                    st.session_state.processed_filenames.extend(new_filenames)
                    save_processed_files(st.session_state.processed_hashes, st.session_state.processed_filenames)
                    st.success("Documents refreshed and indexed!")
                else:
                    st.info("No new documents to process.")
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
    
    with col2:
        uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded_file is not None:
            try:
                if uploaded_file.name not in st.session_state.processed_filenames:
                    with open(PDF_FOLDER / uploaded_file.name, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.session_state.processed_filenames.append(uploaded_file.name)
                    st.success("File uploaded successfully!")
                else:
                    st.warning("File already exists in processed documents.")
            except Exception as e:
                st.error(f"Error uploading file: {str(e)}")
    
    
    
    # Display list of processed documents
    st.subheader("Processed Documents")
    if st.session_state.processed_filenames:
        with st.expander("PDF Documents", expanded=True):
            for filename in st.session_state.processed_filenames:
                st.write(filename)
    else:
        st.info("No documents have been processed yet.")

def show_chat_history():
    """Display chat history interface"""
    st.markdown('<h1>Chat History</h1>', unsafe_allow_html=True)
    
    if 'message_history' not in st.session_state or not st.session_state.message_history:
        st.info("No chat history yet")
        return
    
    with st.container():
        st.markdown('<div class="chat-history">', unsafe_allow_html=True)
        for i, (sender, message) in enumerate(reversed(st.session_state.message_history)):
            st.markdown(f'<p><strong>{sender}:</strong> {message}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()