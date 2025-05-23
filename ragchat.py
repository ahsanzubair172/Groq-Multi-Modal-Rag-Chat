from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
# from fastapi.responses import JSONResponse
import uvicorn

# Import your existing components
from rag import (
    llm,
    embed_model,
    index,
    memory,
    Settings,
    ChatMemoryBuffer
)

app = FastAPI()

# Input model
class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None  # For multi-user support if needed

# Initialize shared memory buffer
memory = ChatMemoryBuffer(token_limit=2500)

# Initialize settings
Settings.llm = llm
Settings.embed_model = embed_model

# Enhanced chat engine initialization
def get_chat_engine():
    return index.as_chat_engine(
        system_prompt=(
            "You are a helpful, knowledgeable, and friendly AI assistant. "
            "When answering a question, first answer using ONLY the provided context from the documents. "
            "If you have additional relevant information from your own general knowledge, add it as a separate section after the document-based answer, clearly labeled as 'Additional Information from AI'. "
            "If the answer is not in the context, say 'I don't know based on the provided documents.' before providing any additional information."
        ),
        memory=memory,
    )

# Create a singleton chat engine instance
chat_engine = get_chat_engine()


@app.get("/")
async def root():
    return {"mess..": "Welcome to the rag chatbot!"}


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    try:
        response = chat_engine.chat(request.query)
        response_text = str(response)
        # response_text = response.txt
        return {"response": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/health")
# async def health_check():
#     return {"status": "healthy"}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)