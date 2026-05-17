import os
import uuid
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process
from fastapi.responses import StreamingResponse
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NOTE: Ensure you set your OpenAI API key in your environment
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

app = FastAPI(
    title="Agentic Content Creator API", 
    description="An asynchronous CrewAI API powered by FastAPI."
)

# --- Pydantic Models ---
class ContentRequest(BaseModel):
    topic: str
    max_iterations: Optional[int] = 3

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[str] = None

# --- In-Memory Database ---
# In a real-world app, use Redis or Postgres to track task statuses
task_db: Dict[str, Dict[str, Any]] = {}

# --- CrewAI Logic ---
def create_content_crew(topic: str, max_iterations: int) -> Crew:
    """Instantiate the Agents and Tasks for the specific topic."""
    
    researcher = Agent(
        role='Senior Tech Researcher',
        goal=f'Conduct thorough research on {topic}',
        backstory='An expert analyst who uncovers deep insights and factual data.',
        verbose=True,
        allow_delegation=False
    )
    
    writer = Agent(
        role='Content Strategist',
        goal=f'Write a compelling, easy-to-read article about {topic}',
        backstory='A creative writer who turns complex research into engaging markdown articles.',
        verbose=True,
        allow_delegation=False
    )
    
    research_task = Task(
        description=f"Research {topic}. Gather key bullet points, pros and cons, and industry context.",
        expected_output="A detailed summary of factual research.",
        agent=researcher,
    )
    
    write_task = Task(
        description=f"Using the research provided, write an engaging article about {topic}.",
        expected_output="A polished 500-word article in markdown format.",
        agent=writer,
    )
    
    return Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
    )

def run_crew_background(task_id: str, topic: str, max_iterations: int):
    """The function that runs in the background to prevent API timeouts."""
    try:
        logger.info(f"Starting Crew execution for task: {task_id}")
        crew = create_content_crew(topic, max_iterations)
        
        # .kickoff() is synchronous and takes time
        result = crew.kickoff() 
        
        # Save the result string (CrewOutput usually casts to str)
        task_db[task_id]["status"] = "completed"
        task_db[task_id]["result"] = str(result)
        logger.info(f"Task {task_id} completed successfully.")
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")
        task_db[task_id]["status"] = "failed"
        task_db[task_id]["result"] = str(e)


# --- API Endpoints ---
@app.post("/generate-content", response_model=TaskStatusResponse)
async def generate_content(request: ContentRequest, background_tasks: BackgroundTasks):
    """
    Endpoint to trigger the agentic workflow. 
    Returns immediately with a task_id while agents work in the background.
    """
    task_id = str(uuid.uuid4())
    
    # Initialize the database entry
    task_db[task_id] = {
        "status": "processing",
        "result": None
    }
    
    # Send the heavy lifting to the background
    background_tasks.add_task(run_crew_background, task_id, request.topic, request.max_iterations)
    
    return TaskStatusResponse(task_id=task_id, status="processing")

@app.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_status(task_id: str):
    """Poll this endpoint to check if the AI is done and get the results."""
    if task_id not in task_db:
        raise HTTPException(status_code=404, detail="Task ID not found")
        
    data = task_db[task_id]
    return TaskStatusResponse(
        task_id=task_id,
        status=data["status"],
        result=data["result"]
    )
