import os
from llama_index.core import (
    TreeIndex,
    VectorStoreIndex,
    ListIndex,
    KeywordTableIndex,
    ComposableGraph,
    SimpleDirectoryReader,
    Settings
)
from llama_index.core import Document
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from API import MyApi

# Suppress TensorFlow logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

# Initialize LLM
print("Initializing LLM...")
llm = Groq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=MyApi,
    max_tokens=600,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    num_beams=1,
    do_sample=False,
)
Settings.llm = llm

# Initialize embedding model
print("Initializing embeddings...")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

# Initialize chat memory
memory = ChatMemoryBuffer(token_limit=2500)

# Initialize node parser
print("Initializing node parser...")
parser = SentenceWindowNodeParser.from_defaults(
    window_size=5, # 5 sentences for a chunks 2 before and 2 after the current sentence
    include_metadata=True, # Include metadata in the nodes
    include_prev_next_rel=True # Include previous and next relationships in the metadata
)

# Load documents
print("Loading documents...")
documents = SimpleDirectoryReader(r"D:\Ahsan\Office\check").load_data()



# Split documents into categories based on their content
general_docs = [doc for doc in documents if doc.metadata['file_name'] in ['general.txt', 'general1.txt']]
keyword_docs = [doc for doc in documents if doc.metadata['file_name'] in ['keywords.txt', 'keywords1.txt']]
report_docs = [doc for doc in documents if doc.metadata['file_name'] in ['report_summary.txt']]



# Parse nodes for each category
print("Parsing nodes...")
general_nodes = parser.get_nodes_from_documents(general_docs)
keyword_nodes = parser.get_nodes_from_documents(keyword_docs)
report_nodes = parser.get_nodes_from_documents(report_docs)

# Create different indexess
print("Creating indexes...")
vector_index = VectorStoreIndex(general_nodes)
tree_index = TreeIndex(report_nodes)
keyword_index = KeywordTableIndex(keyword_nodes)

# Create graph index
index_summaries = [
    "Vector index for general knowledge questions.",
    "Tree index for hierarchical document search.",
    "keyword index for sequential data access."
]

print("Building graph index...")
graph = ComposableGraph.from_indices(
    TreeIndex,
    [vector_index, keyword_index, tree_index],
    index_summaries=index_summaries,
)

# Create chat engine with system prompt + memory
SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable, and friendly AI assistant. "
    "When answering a question, first answer using ONLY the provided context from the documents. "
    "If you have additional relevant information from your own general knowledge, add it as a separate section "
    "after the document-based answer, clearly labeled as 'Additional Information from AI'. "
    "If the answer is not in the context, say 'I don't know based on the provided documents.' before providing any additional information."
)
chat_engine = graph.as_chat_engine(system_prompt=SYSTEM_PROMPT, memory=memory)

print("Chatbot is ready! Type 'exit'|'quit'|'close'|'bye' to end the conversation.")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ['exit', 'quit', 'bye', 'close']:
        break
    response = chat_engine.chat(user_input)
    print(f"Bot: {str(response)}\n")