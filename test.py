from fastapi import FastAPI
import nest_asyncio
nest_asyncio.apply()
from pydantic import BaseModel
app = FastAPI()

class User(BaseModel):
    name: str
    gender: str
    age: int
    email: str ="-----@gmail.com"
    job: str
    city: str = "LAHORE"
    country: str = "Pakistan"
    bodyCount: int = 0
    

@app.post("/create_user")
def create_user(user: User):
    return {"user_data": user}


class UserResponse(BaseModel):
    name: str
    email: str

@app.get("/user", response_model=UserResponse)
def get_user():
    return {"name": "Ahsan", "email": "ahsan@example.com", "extra_field": "ignored"}# This extra field will be ignored in the response

