
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from plumber1 import chat_engine  # Import pre-initialized chat engine

app = FastAPI()


class QueryInput(BaseModel):
    query: str

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG chatbot!"}

@app.post("/ask")
async def ask_question(data: QueryInput):
    try:
        response = chat_engine.chat(data.query)
        print("Response:", response)  # Log the response for debugging
        print("Response type:", type(response))  # Log the type of response for debugging
        
        return {"response": str(response)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=4000, reload=True)