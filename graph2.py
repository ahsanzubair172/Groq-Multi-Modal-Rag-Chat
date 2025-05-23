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
from llama_index.core.node_parser import SentenceWindowNodeParser
 # Memory not used with query_engine
 
# from llama_index.core.memory import ChatMemoryBuffer 
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
    max_tokens=300,
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

# Initialize node parser
print("Initializing node parser...")
parser = SentenceWindowNodeParser.from_defaults(
    window_size=5,
    include_metadata=True,
    include_prev_next_rel=True
)

# Load documents
print("Loading documents...")
documents = SimpleDirectoryReader(r"D:\Ahsan\Office\check").load_data()
print(f"Total documents loaded: {len(documents)}")

# Split documents into categories
general_docs = [doc for doc in documents if doc.metadata['file_name'] in ['general.txt', 'general1.txt']]
keyword_docs = [doc for doc in documents if doc.metadata['file_name'] in ['keywords.txt', 'keywords1.txt']]
report_docs = [doc for doc in documents if doc.metadata['file_name'] in ['report_summary.txt']]

print("Documents by category:")
print(f"  - General: {len(general_docs)}")
print(f"  - Keyword: {len(keyword_docs)}")
print(f"  - Report: {len(report_docs)}")

# Parse nodes
print("Parsing nodes...")
general_nodes = parser.get_nodes_from_documents(general_docs)
keyword_nodes = parser.get_nodes_from_documents(keyword_docs)
report_nodes = parser.get_nodes_from_documents(report_docs)

print("Nodes parsed:")
print(f"  - General: {len(general_nodes)}")
print(f"  - Keyword: {len(keyword_nodes)}")
print(f"  - Report: {len(report_nodes)}")

# Create indexes
print("Creating indexes...")
vector_index = VectorStoreIndex(general_nodes)
print("  Vector index created.")
tree_index = TreeIndex(report_nodes)
print("  Tree index created.")
keyword_index = KeywordTableIndex(keyword_nodes)
print("  Keyword index created.")

# Create graph index
print("Building composable graph...")
index_summaries = [
    "Vector index for general knowledge questions.",
    "Tree index for hierarchical document search.",
    "Keyword index for keyword-based lookups."
]

graph = ComposableGraph.from_indices(
    # TreeIndex,
    VectorStoreIndex,
    [vector_index, keyword_index, tree_index],
    index_summaries=index_summaries,
)
print("Graph index built.")

# SYSTEM_PROMPT = (
#     "You are a helpful, knowledgeable, and friendly AI assistant."
# )

# memory = ChatMemoryBuffer(token_limit=2500)

# query_engine = graph.as_query_engine(
#     system_prompt=SYSTEM_PROMPT,
#     memory=memory
# )

# Since memory and system_prompt are not supported by `as_query_engine`, use without them
query_engine = graph.as_query_engine()

print("\nChatbot is ready!")
print("Type 'exit'|'quit'|'close'|'bye' to end the conversation.\n")

# Chat loop
while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ['exit', 'quit', 'bye', 'close']:
        print("Goodbye!")
        break
    response = query_engine.query(user_input)
    print(f"Bot: {str(response)}\n")
