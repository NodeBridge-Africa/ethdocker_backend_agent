from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import AsyncOpenAI
import httpx
import sys
import os
import logging

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart
)

from ethdocker_expert import ethdocker_expert, EthDockerDeps

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="ETHDocker Expert API")
security = HTTPBearer()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Request/Response Models
class ETHDockerRequest(BaseModel):
    query: str
    user_id: str
    request_id: str
    session_id: str
    client_info: Optional[Dict[str, Any]] = None

class ETHDockerResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    error: Optional[str] = None

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
    """Verify the bearer token against environment variable."""
    expected_token = os.getenv("API_BEARER_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=500,
            detail="API_BEARER_TOKEN environment variable not set"
        )
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return True

async def fetch_conversation_history(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch the most recent conversation history for a session."""
    try:
        response = supabase.table("ethdocker_messages") \
            .select("*") \
            .eq("session_id", session_id) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        
        # Convert to list and reverse to get chronological order
        messages = response.data[::-1]
        return messages
    except Exception as e:
        logger.error(f"Failed to fetch conversation history: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to fetch conversation history: {str(e)}"
        )

async def store_message(
    session_id: str,
    message_type: str,
    content: str,
    data: Optional[Dict] = None
):
    """Store a message in the Supabase ethdocker_messages table."""
    try:
        message_obj = {
            "type": message_type,
            "content": content,
            "data": data or {}
        }

        supabase.table("ethdocker_messages").insert({
            "session_id": session_id,
            "message": message_obj
        }).execute()
    except Exception as e:
        logger.error(f"Failed to store message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store message: {str(e)}"
        )

@app.post("/api/ethdocker-expert", response_model=ETHDockerResponse)
async def ethdocker_expert_endpoint(
    request: ETHDockerRequest,
    authenticated: bool = Depends(verify_token)
):
    """
    ETHDocker expert endpoint that handles:
    - Conversation history
    - Message storage
    - Error handling
    - Client information tracking
    """
    try:
        # Log request
        logger.info(f"Processing request {request.request_id} for session {request.session_id}")
        
        # Fetch conversation history
        conversation_history = await fetch_conversation_history(request.session_id)
        
        # Convert conversation history to format expected by agent
        messages = []
        for msg in conversation_history:
            msg_data = msg["message"]
            msg_type = msg_data["type"]
            msg_content = msg_data["content"]
            msg = (
                ModelRequest(parts=[UserPromptPart(content=msg_content)])
                if msg_type == "human"
                else ModelResponse(parts=[TextPart(content=msg_content)])
            )
            messages.append(msg)

        # Store user's query
        await store_message(
            session_id=request.session_id,
            message_type="human",
            content=request.query,
            data={
                "user_id": request.user_id,
                "request_id": request.request_id,
                "client_info": request.client_info
            }
        )

        # Initialize agent dependencies
        deps = EthDockerDeps(
            supabase=supabase,
            openai_client=openai_client
        )

        # Run the ETHDocker expert
        result = await ethdocker_expert.run(
            request.query,
            message_history=messages,
            deps=deps
        )

        # Store agent's response
        await store_message(
            session_id=request.session_id,
            message_type="ai",
            content=result.data,
            data={
                "request_id": request.request_id,
                "user_id": request.user_id
            }
        )

        return ETHDockerResponse(
            success=True,
            message="Successfully processed request"
        )

    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        logger.error(error_msg)
        
        # Store error message in conversation
        await store_message(
            session_id=request.session_id,
            message_type="error",
            content="I apologize, but I encountered an error processing your request.",
            data={
                "error": str(e),
                "request_id": request.request_id,
                "user_id": request.user_id
            }
        )
        
        return ETHDockerResponse(
            success=False,
            error=error_msg
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port) 