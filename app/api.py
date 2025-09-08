"""
FastAPI application for the shopping agent API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from .shopping_agent import (
    main as initialize_agent,
    get_structured_product_recommendations,
    get_structured_image_search_results
)

app = FastAPI(
    title="Shopping Agent API",
    description="AI-powered shopping assistant with product search and image recognition",
    version="1.0.0"
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize the shopping agent on startup"""
    global agent
    try:
        agent = initialize_agent()
        print("Shopping agent initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize shopping agent: {e}")
        raise e

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    image_search: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    products: List[dict]

# Structured Product Models for Frontend Cards
# class ProductCard(BaseModel):
#     """Individual product card for frontend display"""
#     id: str  # Unique identifier
#     title: str
#     price: float
#     rating: float
#     category: str
#     image_url: str
#     product_url: str
#     is_bestseller: bool
#     sales_volume: int
#     tier: str  # "premium" or "value"
#     tier_label: str  # "Premium Choice" or "Value Alternative"
#     description: Optional[str] = None


# API Endpoints

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    General chat endpoint that can handle any user query.
    The agent will automatically route to appropriate tools.
    """
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        result = agent.invoke({"input": request.message})
        response = result.get("output", "Sorry, I couldn't process your request.")
        return ChatResponse(
            response=response,
            session_id=request.session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# @app.post("/search/products", response_model=ChatResponse)
# async def search_products(request: ChatRequest):
#     """
#     Search for products based on text query - returns structured data for frontend cards
#     """
#     try:
#         # Use structured product recommendations
#         result = get_structured_product_recommendations(request.query)
        
#         # Convert to response model
#         product_cards = [ProductCard(**product) for product in result["products"]]
        
#         return ChatResponse(
#             products=product_cards,
#             session_id=request.session_id
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error searching products: {str(e)}")

# @app.post("/search/image", response_model=ChatResponse)
# async def search_by_image(request: ChatRequest):
#     """
#     Search for products similar to test image in data folder (test mode)
#     Returns structured data for frontend cards
#     """
#     try:
#         # Use structured image search
#         result = get_structured_image_search_results()
        
#         # Convert to response model
#         product_cards = [ProductCard(**product) for product in result["products"]]
        
#         return ChatResponse(
#             products=product_cards,
#             session_id=request.session_id
#         )
            
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing image search: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring
    """
    return {
        "status": "healthy",
        "agent_initialized": agent is not None,
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
