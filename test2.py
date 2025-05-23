import os
import requests
import pinecone
from sentence_transformers import SentenceTransformer
import pdfplumber
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY must be set in .env file.")
PINECONE_ENV = os.getenv('PINECONE_ENV')
if not PINECONE_ENV:
    raise ValueError("PINECONE_ENV must be set in .env file.")
PINECONE_INDEX = "myself"
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY must be set in .env file.")
PDF_FOLDER = r"D:\Ahsan\Office\Data"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5
MAX_TOKENS = 2048

# Initialize components
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len
)

def process_pdfs(folder):
    """Process all PDFs in a folder and store embeddings"""
    if not pinecone.list_indexes():
        pinecone.create_index(
            PINECONE_INDEX,
            dimension=384,
            metric='dotproduct',
            pod_type='p1'
        )
    
    index = pinecone.Index(PINECONE_INDEX)

    for filename in os.listdir(folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder, filename)
            try:
                with pdfplumber.open(file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                    chunks = text_splitter.split_text(text)
                    
                    vectors = []
                    for i, chunk in enumerate(chunks):
                        embedding = embedding_model.encode(chunk)
                        vectors.append({
                            'id': f"{filename}-{i}",
                            'values': embedding.tolist(),
                            'metadata': {'source': filename}
                        })
                    
                    index.upsert(vectors=vectors)
                    print(f"Processed {filename} with {len(chunks)} chunks")
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

def get_relevant_context(query):
    """Retrieve relevant context from Pinecone"""
    index = pinecone.Index(PINECONE_INDEX)
    query_embedding = embedding_model.encode(query).tolist()
    
    results = index.query(
        vector=query_embedding,
        top_k=TOP_K,
        include_metadata=True
    )
    
    context = []
    for match in results['matches']:
        context.append({
            'text': match['id'].split('-')[0][:100],
            'content': match['values'],
            'score': match['score']
        })
    
    return context

def generate_answer(question, context):
    """Generate answer using Groq API"""
    prompt = f"""
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
    
    headers = {
        'Authorization': f'Bearer {GROQ_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS
    }
    
    try:
        response = requests.post(
            "https://api.groq.ai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        return f"Error generating answer: {str(e)}"

def main():
    print("Processing PDFs...")
    process_pdfs(PDF_FOLDER)
    
    print("Chatbot ready. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
            
        context = get_relevant_context(user_input)
        formatted_context = "\n".join([f"Doc {i+1}: {c['text']}..." for i, c in enumerate(context)])
        
        answer = generate_answer(user_input, formatted_context)
        print(f"\nAssistant: {answer}")

if __name__ == "__main__":
    main()