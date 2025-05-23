import os
from PyPDF2 import PdfReader
from llama_index.core import Document  # Import the Document class
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

# Specify the PDF file path
file_path = r"D:\Ahsan\Office\Data\Tasks Report.pdf"

# Function to load PDF content using PyPDF2
def load_pdf_with_pypdf2(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    # Return a list of Document objects instead of dictionaries
    return [Document(text=text, metadata={"file_name": os.path.basename(file_path)})]

# Load the PDF document
documents = load_pdf_with_pypdf2(file_path)

# Explicitly chunk documents before indexing
parser = TokenTextSplitter.from_defaults(chunk_size=250, chunk_overlap=50)
nodes = parser.get_nodes_from_documents(documents)
# Build the index
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
        response = chat_engine.chat(user_input)
        print(f"Chatbot: {response}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")