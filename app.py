from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Iterator, Optional
import json
import asyncio
import os
import logging
import toml
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
env_path = Path('.') / '.env'
if env_path.is_file():
    load_dotenv(dotenv_path=env_path)
    logger.info("Loaded environment variables from .env file")

# Contact information constants
SUPPORT_EMAIL = "contactus@myayurhealth.com"
SUPPORT_PHONE = "+1-612-203-7355"

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
            
            # Check if collection exists, if not create it
            collections = self.client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            
            if self.collection_name not in collection_names:
                # For Qdrant 1.15.1, we need to use the full configuration
                from qdrant_client.http.models import VectorParams, Distance
                
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "text": VectorParams(
                            size=384,  # all-MiniLM-L6-v2 uses 384 dimensions
                            distance=Distance.COSINE
                        )
                    }
                )
                logger.info(f"Created new collection: {self.collection_name}")
            
            logger.info("Vector DB initialized successfully")
        except Exception as e:
            self.client = None
            self.model = None
            logger.error(f"Vector DB Initialization Error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Vector DB Initialization Error: {str(e)}"
            )
    
    def search(self, query: str, limit: int = 5) -> List[DocumentResponse]:
        if not self.client or not self.model:
            return []
        
        try:
            query_vector = self.model.encode(query).tolist()
            
            # First check if collection exists and has points
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            if collection_info.points_count == 0:
                logger.warning(f"Collection '{self.collection_name}' is empty. No documents to search.")
                return []
                
            # Perform the search using the correct API
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            
            return [
                DocumentResponse(
                    content=hit.payload.get('text', ''),
                    confidence=float(hit.score) if hasattr(hit, 'score') else 0.0,
                    metadata=hit.payload.get('metadata', {}),
                    is_doctor_info='doctor' in hit.payload.get('metadata', {}).get('type', '').lower()
                )
                for hit in search_result
            ]
        except Exception as e:
            logger.error(f"Search Error: {str(e)}")
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
        
        groq_api_key = config.get("GROQ_API_KEY")
        if not groq_api_key:
            logger.warning("GROQ_API_KEY not set. Some features may be limited.")
            self.model = None
        else:
            try:
                self.model = Agent(
                    model=Groq(
                        id="llama-3.3-70b-versatile",
                        api_key=groq_api_key
                    ),
                    stream=True,
                    description="Expert Ayurvedic healthcare assistant",
                    instructions=[
                        "Provide accurate Ayurvedic information based on available documentation",
                        "Only recommend doctors that are explicitly mentioned in the documentation",
                        "For health issues, explain Ayurvedic treatment approaches and recommend relevant doctors",
                        "Be clear when information comes from documentation versus general knowledge"
                    ]
                )
                logger.info("Ayurveda Expert System initialized with Groq model")
            except Exception as e:
                logger.error(f"Failed to initialize Groq model: {str(e)}")
                self.model = None
    
    async def process_query(self, query: str) -> Tuple[str, List[DocumentResponse]]:
        logger.info(f"Processing query: {query}")
        try:
            if any(keyword in query.lower() for keyword in ['doctor', 'practitioner', 'physician', 'vaidya']):
                return await self.process_doctor_query(query)
            elif any(keyword in query.lower() for keyword in ['treat', 'cure', 'healing', 'medicine', 'therapy', 'disease', 'condition', 'problem', 'pain']):
                return await self.process_health_query(query)
            else:
                return await self.process_general_query(query)
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

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
        logger.info(f"Generated prompt: {prompt[:200]}...")
        
        if not self.model:
            return ("I'm currently operating in limited mode. To get full AI-powered responses, "
                   f"please set up your GROQ_API_KEY. Contact {SUPPORT_EMAIL} for assistance.")
        
        try:
            response = await self.model.arun(prompt)
            return response.content
        except Exception as e:
            logger.error(f"Model error: {str(e)}")
            return ("I'm having trouble connecting to the AI service. "
                   f"Please try again later or contact {SUPPORT_EMAIL} for assistance.")

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
    # First try to load from environment variables (which can be set in .env file)
    config = {
        "QDRANT_URL": os.getenv("QDRANT_URL", "https://5b0cccfe-e220-49b7-8f69-2bf1a5a7b7f6.europe-west3-0.gcp.cloud.qdrant.io"),
        "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY", ""),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY", "")
    }
    
    # Log which keys are set (without showing the actual values for security)
    for key in config:
        if config[key]:
            logger.info(f"Found {key} in environment")
        else:
            logger.warning(f"{key} not found in environment")
    
    # Fallback to secrets.toml if needed (for backward compatibility)
    if not all(config.values()):
        try:
            with open("secrets.toml", "r") as f:
                toml_config = toml.load(f)
                # Only update keys that aren't already set
                for key, value in toml_config.items():
                    if key in config and not config[key]:
                        config[key] = value
                logger.info("Loaded missing config from secrets.toml")
        except FileNotFoundError:
            logger.warning("secrets.toml not found, using environment variables only")
    
    return config

# Initialize expert system
expert_system = AyurvedaExpertSystem(load_config())

def create_sse_message(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"

async def stream_response(response: str) -> Iterator[str]:
    for word in response.split():
        yield create_sse_message({"token": word + " "})
        await asyncio.sleep(0.05)

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask/stream")
async def stream_chat(question: Question):
    logger.info(f"Received question: {question.question}")
    try:
        response, docs = await expert_system.process_query(question.question)
        logger.info(f"Generated response (first 200 chars): {response[:200]}...")
        
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
        
        return StreamingResponse(
            stream_response(response),
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"Error in stream_chat: {str(e)}")
        error_msg = create_sse_message({
            "error": f"Error processing query: {str(e)}"
        })
        return StreamingResponse(
            iter([error_msg]),
            headers={"Content-Type": "text/event-stream"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000, log_level="info")
