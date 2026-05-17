# from fastapi import FastAPI
 
# from enum import Enum
 
# app = FastAPI()

# @app.get("/hello/{name}")
# async def hello(name):
#     return f"Welcome {name}"
 
# # @app.get("/hello")
# # async def hello():
#     # return "Welcome to FastAPI"

from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
 
# 1. Initialize FastAPI
app = FastAPI(title="Simple GenAI API")
 
# 2. Initialize the Gemini Client with your hardcoded key
GEMINI_API_KEY = "AIzaSyBorNAJN5Mg68ClRw2AUGQy-qbc8o0GmJM"  # Replace with your actual key
client = genai.Client(api_key=GEMINI_API_KEY)
 
# 3. Define what the incoming request data should look like
class QueryRequest(BaseModel):
    prompt: str
 
# 4. Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check to verify the service is running."""
    return {"status": "ok"}
 
# 5. Create the API endpoint
@app.post("/ask")
async def ask_gemini(request: QueryRequest):
    """
    Takes a prompt, waits for Gemini to finish generating the entire
    response, and returns it as a standard JSON object.
    """
    # This matches your simple code exactly
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=request.prompt,
    )
    
    # Return the text back to the client as JSON
    return {"response": response.text}