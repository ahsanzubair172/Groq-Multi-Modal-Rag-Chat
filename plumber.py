import os
import pdfplumber
from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core import KeywordTableIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from API import MyApi

# Suppress TensorFlow and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

# Initialize LLM
llm = Groq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    api_key=MyApi,
    max_tokens=700,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    num_beams=4,
    do_sample=True,
)

# Initialize settings
Settings.llm = llm

# Initialize local embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

# Initialize chat memory buffer with token limit
memory = ChatMemoryBuffer(token_limit=2500)

# Specify the folder or file path
folder_path = r"D:\Ahsan\Office\Data"

# Function to load a single PDF with pdfplumber
def load_pdf_with_pdfplumber(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return [Document(text=text, metadata={"file_name": os.path.basename(file_path)})]

# Function to load all PDFs in a folder with pdfplumber
def load_pdfs_from_folder(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            documents.extend(load_pdf_with_pdfplumber(file_path))
    return documents


# Load all PDFs from the specified folder
documents = load_pdfs_from_folder(folder_path)

# Check if all PDFs in the folder were loaded successfully
pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
if len(documents) == len(pdf_files):
    print(f"All {len(pdf_files)} PDF files loaded successfully.")
else:
    print(f"Warning: Found {len(pdf_files)} PDF files, but only loaded {len(documents)}. Some files may have failed to load.")



# Explicitly chunk documents before indexing
parser = TokenTextSplitter.from_defaults(chunk_size=500, chunk_overlap=80)
nodes = parser.get_nodes_from_documents(documents)
# Build the index
print("indexing .....")
index = KeywordTableIndex(nodes)

# Create a simple chat engine with context and memory
chat_engine = index.as_chat_engine(
    system_prompt=(
        "You are a helpful, knowledgeable, and friendly AI assistant. "
        "When answering a question, first answer using ONLY the provided context from the documents. "
        "If you have additional relevant information from your own general knowledge, add it as a separate section after the document-based answer, clearly labeled as 'Additional Information from AI'. "
        "If the answer is not in the context, say 'I don't know based on the provided documents.' before providing any additional information."
    ),
    memory=memory,
)

print("Chatbot is ready! Type 'exit'|'quit'|'close'|'bye' to end the conversation.")

while True:
    user_input = input("You: ")
    user_input = user_input.strip()
    
    if user_input.lower() in ["exit", "quit", "close", "bye"]:
        print("Chatbot: Goodbye!")
        break
    
    try:
        # Get the response using the chat engine
        print("chat_engine",chat_engine)
        response = chat_engine.chat(user_input)
        print(f"Chatbot: {response}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
        
        
        
