from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_index.llms.groq import Groq
from API import MyApi

# Init LLM
llm = Groq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    api_key=MyApi,
    max_tokens=1024,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    num_beams=4,
    do_sample=True
)

# FastAPI app
app = FastAPI()

@app.get("/")
def Welcome():
    return {"message": "Welcome to the Groq Chat API!"}

# Input model
class query(BaseModel):
    you: str

@app.post("/chat")
def chat(msg: query):
    user_input = msg.you.strip()
    if user_input.lower() in ["exit", "quit", "bye"]:
        return {"response": "Goodbye!"}
        
    
    try:
        result = llm.complete(user_input)
        # print("response", result)# to check
        # print("Result Type:", type(result))# to check result type
        return {"response": str(result)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
