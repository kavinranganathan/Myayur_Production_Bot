from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from pydantic import BaseModel
from typing import Iterator, Optional
import json
import asyncio
import os
import toml
from typing import List, Dict, Tuple
from dataclasses import dataclass
from phi.agent import Agent
from phi.model.groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Contact information constants
SUPPORT_EMAIL = "support@myayurhealth.com"
SUPPORT_PHONE = "+1 (555) 123-4567"

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

@dataclass
class DocumentResponse:
    content: str
    confidence: float
    metadata: Dict
    is_doctor_info: bool = False

class Question(BaseModel):
    question: str

class VectorDBService:
    def __init__(self, api_url: str = None, api_key: str = None):
        try:
            if api_url and api_key:
                self.client = QdrantClient(url=api_url, api_key=api_key)
            else:
                self.client = QdrantClient(":memory:")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.collection_name = "myayurhealth_docs"
        except Exception as e:
            self.client = None
            self.model = None
            raise HTTPException(
                status_code=500,
                detail=f"Vector DB Initialization Error: {str(e)}"
            )
    
    def search(self, query: str, limit: int = 5) -> List[DocumentResponse]:
        if not self.client or not self.model:
            return []
        
        try:
            query_vector = self.model.encode(query).tolist()
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            
            return [
                DocumentResponse(
                    content=result.payload.get('text', ''),
                    confidence=float(result.score),
                    metadata=result.payload.get('metadata', {}),
                    is_doctor_info='doctor' in result.payload.get('metadata', {}).get('type', '').lower()
                )
                for result in results
            ]
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Search Error: {str(e)}"
            )

class AyurvedaExpertSystem:
    def __init__(self, config: Dict[str, str]):
        self.vector_db = VectorDBService(
            api_url=config.get("QDRANT_URL"),
            api_key=config.get("QDRANT_API_KEY")
        )
        self.model = Agent(
            model=Groq(id="llama-3.3-70b-versatile"),
            stream=True,
            description="Expert Ayurvedic healthcare assistant",
            instructions=[
                "Provide accurate Ayurvedic information based on available documentation",
                "Only recommend doctors that are explicitly mentioned in the documentation",
                "For health issues, explain Ayurvedic treatment approaches and recommend relevant doctors",
                "Be clear when information comes from documentation versus general knowledge"
            ]
        )
    
    async def process_query(self, query: str) -> Tuple[str, List[DocumentResponse]]:
        # Check if query is about doctors
        if any(keyword in query.lower() for keyword in ['doctor', 'practitioner', 'physician', 'vaidya']):
            return await self.process_doctor_query(query)
        
        # Check if query is about health conditions
        elif any(keyword in query.lower() for keyword in ['treat', 'cure', 'healing', 'medicine', 'therapy', 'disease', 'condition', 'problem', 'pain']):
            return await self.process_health_query(query)
        
        # General query processing
        return await self.process_general_query(query)

    async def process_doctor_query(self, query: str) -> Tuple[str, List[DocumentResponse]]:
        docs = self.vector_db.search(query)
        doctor_docs = [doc for doc in docs if doc.is_doctor_info]
        
        if not doctor_docs:
            return (self.get_no_doctors_message(), [])
        
        context = "\n".join([doc.content for doc in doctor_docs])
        response = await self.get_model_response(context, "doctor")
        return response, doctor_docs

    async def process_health_query(self, query: str) -> Tuple[str, List[DocumentResponse]]:
        condition_docs = self.vector_db.search(query)
        doctor_docs = self.vector_db.search(f"doctor treating {query}")
        doctor_docs = [doc for doc in doctor_docs if doc.is_doctor_info]
        
        all_docs = condition_docs + doctor_docs
        
        if not all_docs:
            response = await self.get_model_response("", "health", query=query)
            return response, []
        
        context = "\n".join([doc.content for doc in all_docs])
        response = await self.get_model_response(context, "health", query=query)
        return response, all_docs

    async def process_general_query(self, query: str) -> Tuple[str, List[DocumentResponse]]:
        docs = self.vector_db.search(query)
        if not docs:
            response = await self.get_model_response("", "general", query=query)
            return response, []
        
        context = "\n".join([doc.content for doc in docs])
        response = await self.get_model_response(context, "general", query=query)
        return response, docs

    async def get_model_response(self, context: str, response_type: str, query: str = "") -> str:
        prompt = self.generate_prompt(context, response_type, query)
        return (await self.model.arun(prompt)).content

    def generate_prompt(self, context: str, response_type: str, query: str = "") -> str:
        base_contact = f"\n\nFor more information and assistance, contact:\nEmail: {SUPPORT_EMAIL}\nPhone: {SUPPORT_PHONE}"
        
        if response_type == "doctor":
            return f"Based on the following doctor information:\n{context}\n\nProvide a clear response listing available doctors with their specializations and qualifications.{base_contact}"
        elif response_type == "health":
            return f"Based on the following information about {query}:\n{context}\n\nProvide a comprehensive response including Ayurvedic treatment approaches and available specialist doctors.{base_contact}"
        else:
            return f"Based on the following information:\n{context}\n\nProvide accurate information about {query} from an Ayurvedic perspective.{base_contact}"

    def get_no_doctors_message(self) -> str:
        return f"I apologize, but I couldn't find any doctors matching your query in our platform. Please try a different search or contact our support team:\nEmail: {SUPPORT_EMAIL}\nPhone: {SUPPORT_PHONE}"

def load_config():
    config = {
        "QDRANT_URL": "http://localhost:6333",
        "QDRANT_API_KEY": ""
    }
    
    # Load from environment variables first
    for key in config:
        env_value = os.getenv(key)
        if env_value:
            config[key] = env_value
    
    # Try to load from secrets.toml if environment variables are missing
    if not config["QDRANT_URL"] or not config["QDRANT_API_KEY"]:
        try:
            with open("secrets.toml", "r") as f:
                toml_config = toml.load(f)
                config.update(toml_config)
        except FileNotFoundError:
            pass
    
    return config

# Initialize expert system
expert_system = AyurvedaExpertSystem(load_config())

def create_sse_message(data: dict) -> str:
    """Format data as SSE message"""
    return f"data: {json.dumps(data)}\n\n"

async def stream_response(response: str) -> Iterator[str]:
    """Stream response token by token with artificial delay"""
    words = response.split()
    for word in words:
        message = create_sse_message({"token": word + " "})
        yield message
        await asyncio.sleep(0.05)

@app.get("/")
async def read_root(request: Request):
    """Serve the chat interface"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask/stream")
async def stream_chat(question: Question):
    """Stream chat responses"""
    try:
        response, docs = await expert_system.process_query(question.question)
        
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        
        return StreamingResponse(
            stream_response(response),
            headers=headers
        )
        
    except Exception as e:
        error_msg = create_sse_message({
            "error": f"Error processing query: {str(e)}"
        })
        return StreamingResponse(
            iter([error_msg]),
            headers={"Content-Type": "text/event-stream"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
