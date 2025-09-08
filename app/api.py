"""
fastapi application for the shopping agent api
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import json
import base64
import uuid
from pathlib import Path

from app.shopping_agent import (
    initialize_agent
)

app = FastAPI(
    title="Shopping Agent API",
    description="AI-powered shopping assistant with product search and image recognition",
    version="1.0.0"
)

# configure cors for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# global agent instance
agent = None

@app.on_event("startup")
async def startup_event():
    """initialize the shopping agent on startup"""
    global agent
    try:
        agent = initialize_agent()
        print("Shopping agent initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize shopping agent: {e}")
        raise e

# request/response models
class ChatRequest(BaseModel):
    message: str
    image_search: Optional[bool] = False
    image_data: Optional[str] = None  # base64 string or data url

class ChatResponse(BaseModel):
    response: str
    products: List[dict]

# structured product models for frontend cards
# class ProductCard(BaseModel):
#     """individual product card for frontend display"""
#     id: str  # unique identifier
#     title: str
#     price: float
#     rating: float
#     category: str
#     image_url: str
#     product_url: str
#     is_bestseller: bool
#     sales_volume: int
#     tier: str  # "premium" or "value"
#     tier_label: str  # "premium choice" or "value alternative"
#     description: Optional[str] = None


# api endpoints

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    general chat endpoint that can handle any user query.
    the agent will automatically route to appropriate tools.
    """
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        # prepare message; if image_search is requested, decode and persist image, and prepend hint
        user_message = request.message or ""

        if request.image_search:
            saved_path: Optional[str] = None
            if request.image_data:
                try:
                    data = request.image_data
                    # handle data url
                    if data.startswith("data:") and ";base64," in data:
                        header, b64 = data.split(",", 1)
                        # try to infer file extension
                        ext = "png"
                        if header.startswith("data:image/"):
                            mime_ext = header.split("/")[1].split(";")[0]
                            ext = mime_ext or ext
                    else:
                        b64 = data
                        ext = "png"

                    raw = base64.b64decode(b64)
                    upload_dir = Path("/tmp/palona_uploads")
                    upload_dir.mkdir(parents=True, exist_ok=True)
                    file_name = f"uploaded_{uuid.uuid4().hex}.{ext}"
                    file_path = upload_dir / file_name
                    with open(file_path, "wb") as f:
                        f.write(raw)
                    saved_path = str(file_path)
                except Exception:
                    saved_path = None

            prefix = "user added an image, use the imagesearch tool."
            if saved_path:
                prefix += f" Image path: {saved_path}"
            user_message = f"{prefix}\n{user_message}".strip()

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_message}]},
            {"configurable": {"thread_id": "1"}}
        )
        # extract the response and any tool outputs (aimessage objects)
        messages = result.get("messages", [])
        response = "Sorry, I couldn't process your request."
        products: List[dict] = []

        if messages:
            last_msg = messages[-1]
            # llm final answer
            response = getattr(last_msg, "content", response)

            # consider only messages from the latest user message onward in this turn
            # find index of the last user message (prefer matching current request.message)
            last_user_index = -1
            for i in range(len(messages) - 1, -1, -1):
                m = messages[i]
                role = getattr(m, "type", None) or getattr(m, "role", None)
                content = getattr(m, "content", None)
                if role in ("user", "human"):
                    last_user_index = i
                    # if content matches current request, stop here
                    if isinstance(content, str) and content == request.message:
                        break
            scan_slice = messages[last_user_index + 1:] if last_user_index >= 0 else messages

            # scan messages in reverse within this slice for the most recent tool output that looks like json
            for m in reversed(scan_slice):
                tool_output = None

                # case 1: attached in additional_kwargs
                if hasattr(m, "additional_kwargs") and isinstance(m.additional_kwargs, dict):
                    tool_output = m.additional_kwargs.get("tool_output")

                # case 2: content is a plain string
                if tool_output is None and hasattr(m, "content") and isinstance(m.content, str):
                    text = m.content.strip()
                    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
                        tool_output = text

                # case 3: content is a list of parts (e.g., langgraph message parts)
                if tool_output is None and hasattr(m, "content") and isinstance(m.content, list):
                    for part in reversed(m.content):
                        if isinstance(part, dict):
                            candidate = part.get("text") or part.get("content") or part.get("value")
                            if isinstance(candidate, str):
                                t = candidate.strip()
                                if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
                                    tool_output = t
                                    break
                        elif isinstance(part, str):
                            t = part.strip()
                            if (t.startswith("{") and t.endswith("}")) or (t.startswith("[") and t.endswith("]")):
                                tool_output = t
                                break

                if tool_output:
                    try:
                        parsed = json.loads(tool_output)
                        if isinstance(parsed, dict) and "products" in parsed and isinstance(parsed["products"], list):
                            products = parsed["products"]
                            break
                        elif isinstance(parsed, list):
                            products = parsed
                            break
                    except Exception:
                        # ignore non-json contents
                        pass

        return ChatResponse(
            response=response,
            products=products
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# @app.post("/search/products", response_model=ChatResponse)
# async def search_products(request: ChatRequest):
#     """
#     search for products based on text query - returns structured data for frontend cards
#     """
#     try:
#         # use structured product recommendations
#         result = get_structured_product_recommendations(request.query)
        
#         # convert to response model
#         product_cards = [ProductCard(**product) for product in result["products"]]
        
#         return ChatResponse(
#             products=product_cards,
#             session_id=request.session_id
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"error searching products: {str(e)}")

# @app.post("/search/image", response_model=ChatResponse)
# async def search_by_image(request: ChatRequest):
#     """
#     search for products similar to test image in data folder (test mode)
#     returns structured data for frontend cards
#     """
#     try:
#         # use structured image search
#         result = get_structured_image_search_results()
        
#         # convert to response model
#         product_cards = [ProductCard(**product) for product in result["products"]]
        
#         return ChatResponse(
#             products=product_cards,
#             session_id=request.session_id
#         )
            
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"error processing image search: {str(e)}")

@app.get("/health")
async def health_check():
    """
    health check endpoint for monitoring
    """
    return {
        "status": "healthy",
        "agent_initialized": agent is not None,
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
