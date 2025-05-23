import os
from llama_index.readers.file import PyMuPDFReader
from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceWindowNodeParser

from llama_index.core import TreeIndex, GPTVectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from API import MyApi

# Suppress TensorFlow logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

#  1) Initialize LLM
print("Initializing LLM...") 
llm = Groq(
    # model="meta-llama/llama-4-maverick-17b-128e-instruct",
    model ="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=MyApi,
    max_tokens=300,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    num_beams=1,
    do_sample=False,
)
Settings.llm = llm

#  Initialize embedding model
print("Initializing embeddings...")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

#  Initialize chat memory       
memory = ChatMemoryBuffer(token_limit=2500)

#   PDF loading with PyMuPDF 
print("Loading PDFs...")
# specify your folder here
folder_path = r"D:\Ahsan\Office\Data"

def load_pdf_with_pymupdf(file_path: str):
    """Load a single PDF into a Document via PyMuPDFReader."""
    loader = PyMuPDFReader()
    documents = loader.load_data(file_path=file_path, metadata=True)
    if documents:
        return documents[0]
    else:
        return Document(text="", metadata={"file_name": os.path.basename(file_path)})

def load_pdfs_from_folder(folder_path: str):
    """Load all PDFs in a folder."""
    docs = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".pdf"):
            full_path = os.path.join(folder_path, fname)
            docs.append(load_pdf_with_pymupdf(full_path))
    return docs


documents = load_pdfs_from_folder(folder_path)

# sanity check
pdf_count = len([f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")])
if len(documents) == pdf_count:
    print(f"All {pdf_count} PDF files loaded successfully.")
else:
    print(f"Warning: Found {pdf_count} PDFs, but loaded {len(documents)} documents.")


#   Chunking via sliding‐window sentence splitter 
#    window_size and overlap are in number of sentences
print("Chunking documents...")
parser = SentenceWindowNodeParser.from_defaults(
    window_size=5,               # 5 sentences (2 before, current, 2 after)    
    include_metadata=True,      # keeps useful info like source, file, line
    include_prev_next_rel=True  # include previous and next sentences as relations
)
nodes = parser.get_nodes_from_documents(documents)

#   Build a TreeIndex
print("Building index...")
index = TreeIndex(nodes)
# index = GPTVectorStoreIndex(nodes)

#  Create chat engine with system prompt + memory 

SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable, and friendly AI assistant. "
    "When answering a question, first answer using ONLY the provided context from the documents. "
    "If you have additional relevant information from your own general knowledge, add it as a separate section "
    "after the document-based answer, clearly labeled as 'Additional Information from AI'. "
    "If the answer is not in the context, say 'I don't know based on the provided documents.' before providing any additional information."
)
chat_engine = index.as_chat_engine(system_prompt=SYSTEM_PROMPT, memory=memory)

print("Chatbot is ready! Type 'exit'|'quit'|'close'|'bye' to end the conversation.")

#   Chat loop 
while True:
    user_input = input("You: ").strip()
    if user_input.lower() in {"exit", "quit", "close", "bye"}:
        print("Chatbot: Goodbye!")
        break
            
            
    try:
        resp = chat_engine.chat(user_input)
        print(f"Chatbot: {resp}")
        # Debug: print raw response type and content if possible
        # if hasattr(resp, 'source_nodes'):
        #     print("Debug: source nodes used for response:")
        #     for node in resp.source_nodes:
        #         print(f"- {node.get_text()[:200]}")  # print first 200 chars of each source node
    except Exception as e:
        if "RateLimitError" in str(e):
            print("Rate limit exceeded. Waiting 30 seconds before retrying...")
            import time
            time.sleep(30)  # wait for 30 seconds before retrying
            continue
        print(f"An error occurred: {e}")

    
    
       
    # try:
    #     # Get the response using the chat engine
    #     response = chat_engine.chat(user_input)
    #     print(f"Chatbot: {response}")
    # except Exception as e:
    #     print(f"An error occurred: {str(e)}")