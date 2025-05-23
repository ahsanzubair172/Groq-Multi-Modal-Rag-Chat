


import os
import pdfplumber
from llama_index.core import Document, Settings
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import KeywordTableIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from API import MyApi
import warnings

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

# Initialize LLM 
print("Initializing LLM...")
llm = Groq(
    # model="meta-llama/llama-4-scout-17b-16e-instruct",
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    api_key=MyApi,
    max_tokens=700,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    num_beams=4,
    do_sample=True,
)
Settings.llm = llm
print("LLM initialized successfully.")

# Initialize embeddings 
print("Initializing embeddings...")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
print("Embeddings initialized successfully.")

# Initialize memory once
print("Initializing chat memory...")
memory = ChatMemoryBuffer(token_limit=2500)
print("Chat memory initialized successfully.")

# Load and process documents 
folder_path = os.getenv("PDF_DATA_PATH", r"D:\Ahsan\Office\Data")  # Use environment variable 
print(f"Loading PDFs from {folder_path}...")

def load_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return [Document(text=text, metadata={"file_name": os.path.basename(file_path)})]

documents = []
for filename in os.listdir(folder_path):
    if filename.lower().endswith('.pdf'):
        file_path = os.path.join(folder_path, filename)
        documents.extend(load_pdf(file_path))
print(f"Loaded {len(documents)} documents successfully.")

# Chunk documents 
print("Chunking documents...")
parser = TokenTextSplitter.from_defaults(chunk_size=500, chunk_overlap=80)
nodes = parser.get_nodes_from_documents(documents)
print("Documents chunked successfully.")

# Build index 
print("Building index...")
index = KeywordTableIndex(nodes)
print("Index built successfully.")

# Create chat engine 
print("Creating chat engine...")
chat_engine = index.as_chat_engine(
    system_prompt=(
        "You are a helpful, knowledgeable, and friendly AI assistant. "
        "When answering a question, first answer using ONLY the provided context from the documents. "
        "If you have additional relevant information from your own general knowledge, add it as a separate section after the document-based answer, clearly labeled as 'Additional Information from AI'. "
        "If the answer is not in the context, say 'I don't know based on the provided documents.' before providing any additional information."
    ),
    memory=memory,
)
print("Chat engine created successfully.")