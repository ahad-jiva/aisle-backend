import os
import math
import hashlib
from dotenv import load_dotenv

from langchain.agents import create_react_agent, AgentExecutor, Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_milvus import Milvus
from langchain_core.embeddings import Embeddings
from typing import List, Dict, Any

from PIL import Image
from pydantic import SecretStr
import torch
from transformers import CLIPProcessor, CLIPModel

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
ZILLIZ_URI = os.getenv("ZILLIZ_URI")

def format_docs(docs):
    """Format documents with smart product information including bestseller status"""
    if not docs:
        return "No products found."
    
    formatted_products = []
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        
        # Extract key information
        title = metadata.get('title', 'Unknown Product')
        stars = metadata.get('stars', 0)
        price = metadata.get('price', 0)
        category = metadata.get('category', 'Unknown Category')
        is_bestseller = metadata.get('isBestSeller', False)
        sales_volume = metadata.get('boughtInLastMonth', 0)
        
        # Format price
        price_str = f"${price}" if price > 0 else "Price not available"
        
        # Format rating with visual indicator
        rating_str = f"{stars} stars" if stars > 0 else "No rating"
        
        # Bestseller and sales indicators
        status_indicators = []
        if is_bestseller:
            status_indicators.append("BESTSELLER")
        
        if sales_volume > 1000:
            status_indicators.append(f"{sales_volume:,} sold this month")
        elif sales_volume > 0:
            status_indicators.append(f"{sales_volume} sold this month")
        
        # Product tier indication
        if is_bestseller:
            tier_info = "Premium Choice"
        else:
            tier_info = "Value Alternative"
        
        # Create smart product description
        product_info = f"""Product {i}: {title}
{tier_info} - {' | '.join(status_indicators) if status_indicators else 'Standard Product'}
Rating: {rating_str}
Price: {price_str}
Category: {category}
Description: {doc.page_content}"""
        
        formatted_products.append(product_info)
    
    return "\n\n" + "\n\n".join(formatted_products)

def docs_to_product_cards(docs) -> List[Dict[str, Any]]:
    """Convert documents to structured product card data for frontend"""
    if not docs:
        return []
    
    product_cards = []
    for doc in docs:
        metadata = doc.metadata
        
        # Extract key information
        title = metadata.get('title', 'Unknown Product')
        stars = float(metadata.get('stars', 0))
        price = float(metadata.get('price', 0))
        category = metadata.get('category', 'Unknown Category')
        is_bestseller = bool(metadata.get('isBestSeller', False))
        sales_volume = int(metadata.get('boughtInLastMonth', 0))
        product_url = metadata.get('productURL', '')
        image_url = metadata.get('imgUrl', '')
        
        # Generate unique ID from title and URL
        product_id = hashlib.md5(f"{title}{product_url}".encode()).hexdigest()[:12]
        
        # Determine tier
        tier = "premium" if is_bestseller else "value"
        tier_label = "Premium Choice" if is_bestseller else "Value Alternative"
        
        # Create product card
        product_card = {
            "id": product_id,
            "title": title,
            "price": price,
            "rating": stars,
            "category": category,
            "image_url": image_url,
            "product_url": product_url,
            "is_bestseller": is_bestseller,
            "sales_volume": sales_volume,
            "tier": tier,
            "tier_label": tier_label,
            "description": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        }
        
        product_cards.append(product_card)
    
    return product_cards

def create_recommendation_summary(product_cards: List[Dict[str, Any]], query: str) -> str:
    """Create a summary of the recommendations for the frontend"""
    if not product_cards:
        return "No products found matching your query."
    
    total_products = len(product_cards)
    premium_count = len([p for p in product_cards if p['tier'] == 'premium'])
    value_count = total_products - premium_count
    avg_rating = sum(p['rating'] for p in product_cards) / total_products if product_cards else 0
    
    summary_parts = [
        f"Found {total_products} recommended products for '{query}'."
    ]
    
    if premium_count > 0 and value_count > 0:
        summary_parts.append(f"Includes {premium_count} premium bestsellers and {value_count} value alternatives.")
    elif premium_count > 0:
        summary_parts.append(f"Featuring {premium_count} premium bestselling products.")
    else:
        summary_parts.append(f"Featuring {value_count} high-value alternatives.")
    
    summary_parts.append(f"Average rating: {avg_rating:.1f} stars.")
    
    return " ".join(summary_parts)

def smart_ranking_score(doc):
    """
    Calculate smart ranking score based on multiple factors
    
    Factors:
    - Star rating (0-5): 40% weight
    - Sales volume (boughtInLastMonth): 30% weight  
    - Bestseller status: 30% weight
    """
    metadata = doc.metadata
    
    # Extract and normalize ratings (0-5 scale)
    try:
        stars = float(metadata.get('stars', 0))
        stars = max(0, min(5, stars))  # Clamp to 0-5 range
    except (ValueError, TypeError):
        stars = 0
    
    # Extract and normalize sales volume (log scale for large numbers)
    try:
        sales = int(metadata.get('boughtInLastMonth', 0))
        sales = max(0, sales)
        # Use log scale to prevent extreme sales from dominating
        normalized_sales = math.log10(sales + 1) / math.log10(10001)  # Normalize to 0-1
    except (ValueError, TypeError):
        normalized_sales = 0
    
    # Bestseller bonus
    is_bestseller = metadata.get('isBestSeller', False)
    bestseller_bonus = 1.0 if is_bestseller else 0.0
    
    # Weighted score calculation
    rating_score = (stars / 5.0) * 0.4  # 40% weight for ratings
    sales_score = normalized_sales * 0.3  # 30% weight for sales volume
    bestseller_score = bestseller_bonus * 0.3  # 30% weight for bestseller status
    
    total_score = rating_score + sales_score + bestseller_score
    
    return total_score

def two_tier_product_retrieval(vectorstore, query: str, k_bestsellers: int = 8, k_alternatives: int = 7, final_count: int = 5):
    """
    smart two-tier product retrieval system
    
    Tier 1: Bestsellers ranked by ratings and sales
    Tier 2: Value alternatives (non-bestsellers) with high ratings and lower prices
    
    Args:
        vectorstore: The Milvus vector store
        query: Search query
        k_bestsellers: Number of bestseller candidates to retrieve
        k_alternatives: Number of alternative candidates to retrieve
        final_count: Final number of products to return
    
    Returns:
        List of recommended products with mix of bestsellers and value alternatives
    """
    
    # Tier 1: Get bestselling products
    try:
        bestseller_filter = 'isBestSeller == true'
        bestsellers = vectorstore.similarity_search(
            query, 
            k=k_bestsellers,
            expr=bestseller_filter
        )
    except Exception as e:
        print(f"Bestseller search failed, falling back to unfiltered: {e}")
        # Fallback if filtering not supported
        all_candidates = vectorstore.similarity_search(query, k=k_bestsellers * 2)
        bestsellers = [doc for doc in all_candidates if doc.metadata.get('isBestSeller', False)][:k_bestsellers]
    
    # Sort bestsellers by smart ranking
    bestsellers_ranked = sorted(bestsellers, key=smart_ranking_score, reverse=True)
    
    # Calculate average price of top bestsellers for value threshold
    bestseller_prices = []
    for doc in bestsellers_ranked[:3]:  # Use top 3 bestsellers for price benchmark
        try:
            price = float(doc.metadata.get('price', 0))
            if price > 0:
                bestseller_prices.append(price)
        except (ValueError, TypeError):
            continue
    
    avg_bestseller_price = sum(bestseller_prices) / len(bestseller_prices) if bestseller_prices else 1000
    price_threshold = avg_bestseller_price * 0.8  # Value alternatives should be 20% cheaper
    
    # Tier 2: Get value alternatives (non-bestsellers with good ratings and lower prices)
    try:
        alternative_filter = 'isBestSeller == false'
        alternatives = vectorstore.similarity_search(
            query,
            k=k_alternatives,
            expr=alternative_filter
        )
    except Exception as e:
        print(f"Alternative search failed, falling back to unfiltered: {e}")
        # Fallback if filtering not supported
        all_candidates = vectorstore.similarity_search(query, k=k_alternatives * 2)
        alternatives = [doc for doc in all_candidates if not doc.metadata.get('isBestSeller', False)][:k_alternatives]
    
    # Filter alternatives by price and rating criteria
    quality_alternatives = []
    for doc in alternatives:
        try:
            price = float(doc.metadata.get('price', 0))
            stars = float(doc.metadata.get('stars', 0))
            
            # Criteria for value alternatives:
            # - Price below threshold OR good rating (4+ stars)
            # - Minimum 3.5 star rating
            if (price < price_threshold or stars >= 4.0) and stars >= 3.5:
                quality_alternatives.append(doc)
        except (ValueError, TypeError):
            # Include if we can't parse price/rating (assume it might be good)
            quality_alternatives.append(doc)
    
    # Sort alternatives by rating and sales
    alternatives_ranked = sorted(quality_alternatives, key=smart_ranking_score, reverse=True)
    
    # Combine results: Mix of bestsellers and alternatives
    # Aim for 60% bestsellers, 40% alternatives in final results
    bestseller_count = min(len(bestsellers_ranked), max(1, int(final_count * 0.6)))
    alternative_count = final_count - bestseller_count
    
    final_recommendations = []
    final_recommendations.extend(bestsellers_ranked[:bestseller_count])
    final_recommendations.extend(alternatives_ranked[:alternative_count])
    
    # If we don't have enough, fill with remaining from either tier
    if len(final_recommendations) < final_count:
        remaining_needed = final_count - len(final_recommendations)
        remaining_bestsellers = bestsellers_ranked[bestseller_count:]
        remaining_alternatives = alternatives_ranked[alternative_count:]
        
        # Combine and sort remaining by score
        remaining = remaining_bestsellers + remaining_alternatives
        remaining_sorted = sorted(remaining, key=smart_ranking_score, reverse=True)
        final_recommendations.extend(remaining_sorted[:remaining_needed])
    
    return final_recommendations[:final_count]

class LocalSentenceTransformerEmbeddings(Embeddings):
    """Local sentence transformer embeddings - same as used in vectordb construction"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with the same model used to build the vector database"""
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()

class CLIPImageEmbeddings(Embeddings):
    """Custom embedding class for CLIP image embeddings"""
    
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed text documents - not used for image embeddings"""
        raise NotImplementedError("This embedding class is for images only")
    
    def embed_query(self, text: str) -> List[float]:
        """Embed text query - not used for image embeddings"""
        raise NotImplementedError("This embedding class is for images only")
    
    def embed_image(self, image_path: str) -> List[float]:
        """Embed a single image and return the embedding"""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.numpy().flatten().tolist()

template = """You are a helpful online store assistant with access to a smart product recommendation system. Use this tool as necessary according to the user's query.

## Context:
{context}

## Question:
{question}

When recommending products:

"""
prompt = ChatPromptTemplate.from_template(template)

def main():

    # load llm
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-pro", google_api_key=SecretStr(GEMINI_API_KEY) if GEMINI_API_KEY else None,
        temperature=0.7
    )

    # vector db setup - use the same embedding model as vectordb construction
    embedding_model = LocalSentenceTransformerEmbeddings()

    vectorstore_text = Milvus(
        embedding_function=embedding_model,
        collection_name="palona_text",
        connection_args={
            "uri": ZILLIZ_URI,
            "token": ZILLIZ_TOKEN
        }
    )

    # Create smart two-tier retrieval function
    def smart_retriever(query: str):
        """smart retriever with bestsellers and value alternatives"""
        smart_docs = two_tier_product_retrieval(
            vectorstore_text, 
            query, 
            k_bestsellers=8, 
            k_alternatives=7, 
            final_count=5
        )
        return smart_docs
    
    qa_chain = (
        {
            "context": lambda x: format_docs(smart_retriever(x)),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # image search tool
    image_embeddings = CLIPImageEmbeddings()

    vectorstore_image = Milvus(
        embedding_function=image_embeddings,
        collection_name="palona_image",
        connection_args={
            "uri": ZILLIZ_URI,
            "token": ZILLIZ_TOKEN
        }
    )

    def image_search_tool(image_path: str) -> str:
        """Find products similar to the provided image using visual similarity"""
        try:
            import os
            
            # Validate image path exists
            if not os.path.exists(image_path):
                return f"Error: Image file not found at path: {image_path}"
            
            print(f"Processing image: {image_path}")
            
            # Embed the provided image
            emb = image_embeddings.embed_image(image_path)
            results = vectorstore_image.similarity_search_by_vector(emb, k=5)
            
            if not results:
                return f"No similar products found for image: {os.path.basename(image_path)}"
            
            # Format image search results with consistent product information
            formatted_results = []
            formatted_results.append(f"Visual search results for: {os.path.basename(image_path)}")
            formatted_results.append("=" * 50)
            
            for i, r in enumerate(results, 1):
                title = r.metadata.get('title', 'Unknown Product')
                price = r.metadata.get('price', 0)
                stars = r.metadata.get('stars', 0)
                category = r.metadata.get('category', 'Unknown')
                is_bestseller = r.metadata.get('isBestSeller', False)
                
                tier = "Premium Choice" if is_bestseller else "Value Alternative"
                price_str = f"${price}" if price > 0 else "Price not available"
                rating_str = f"{stars} stars" if stars > 0 else "No rating"
                
                result_info = f"{i}. {title}\n   {tier} | {rating_str} | {price_str} | {category}"
                formatted_results.append(result_info)
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error processing image: {str(e)}"

    # define langchain tools for model to use
    tools = [
        Tool(
            name = "ProductSearch",
            func=lambda q: qa_chain.invoke(q),
            description="Use this to find recommended products from a catalog when the user asks about items"
        ),
        Tool(
            name = "ImageSearch",
            func=lambda image_path: image_search_tool(image_path),
            description="Use this to find products visually similar to a provided image. Pass the full path to the image file. This searches the product catalog using visual similarity."
        )
    ]

    # Create agent prompt template
    agent_prompt = ChatPromptTemplate.from_template("""
You are a helpful online store assistant. You have access to tools to help users find products and search for similar items using visual search.

TOOLS:
------
You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Some important guidelines:
1. List the suggested products conversationally, in order from most to least recommended. Do not mention to the user that there is a two-tier system. If there are no recommended products, say "I couldn't find any products that match your query."
2. If the user provides an image, use the image search tool to find products similar to the image.
3. Mention star ratings, sales volume, and why each product is recommended
4. Highlight the value proposition of each tier (premium quality vs. cost-effective alternatives), but don't specificaly mention the tiers
5. If the user's request is too vague, feel free to ask for more information. Don't make assumptions about the user's needs. It is better to ask for clarification than to make assumptions. Avoid using the tools until you have a concrete idea of what the user needs.
6. Do not use any emojis or special characters in your response
7. The recommended products will be represented in structured data and rendered on a frontend. This means that your response has to be completely conversational, as the user can already see the product information on the UI.
8. Avoid being overly verbose in your response. Don't say things like "I found some great products for you!". Just say "Here are some products you might like". Be concise and to the point.
9. Be as objective as possible in your response. Don't say things like "I think this product is great" or "This product is perfect for you". Remember, you are a helpful assistant, not a salesperson.
10. There is a fine difference between the user asking you to SEARCH for products, and asking you ABOUT products. Note this difference. Don't perform any product searches if the user isn't asking for it.

New input: {input}
{agent_scratchpad}
""")

    memory = ConversationBufferMemory(memory_key="chat_history")

    # Create the react agent
    agent = create_react_agent(llm, tools, agent_prompt, memory=memory)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3
    )
    
    return agent_executor

# Global variables for structured API access
_vectorstore_text = None
_vectorstore_image = None
_image_embeddings = None

def get_structured_product_recommendations(query: str) -> Dict[str, Any]:
    """
    Get structured product recommendations for API endpoints
    Returns JSON-serializable data for frontend cards
    """
    global _vectorstore_text
    
    if _vectorstore_text is None:
        # Initialize if not already done
        embedding_model = LocalSentenceTransformerEmbeddings()
        _vectorstore_text = Milvus(
            embedding_function=embedding_model,
            collection_name="palona_text",
            connection_args={
                "uri": ZILLIZ_URI,
                "token": ZILLIZ_TOKEN
            }
        )
    
    # Get smart product recommendations
    smart_docs = two_tier_product_retrieval(
        _vectorstore_text, 
        query, 
        k_bestsellers=8, 
        k_alternatives=7, 
        final_count=5
    )
    
    # Convert to structured product cards
    product_cards = docs_to_product_cards(smart_docs)
    recommendation_summary = create_recommendation_summary(product_cards, query)
    
    return {
        "products": product_cards,
        "total_results": len(product_cards),
        "query": query,
        "recommendation_summary": recommendation_summary
    }

def get_structured_image_search_results() -> Dict[str, Any]:
    """
    Get structured image search results for API endpoints
    Returns JSON-serializable data for frontend cards
    """
    global _vectorstore_image, _image_embeddings
    
    if _vectorstore_image is None or _image_embeddings is None:
        # Initialize if not already done
        _image_embeddings = CLIPImageEmbeddings()
        _vectorstore_image = Milvus(
            embedding_function=_image_embeddings,
            collection_name="palona_image",
            connection_args={
                "uri": ZILLIZ_URI,
                "token": ZILLIZ_TOKEN
            }
        )
    
    try:
        # Look for test image in data folder
        import glob
        
        data_folder = "data"
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.webp"]
        
        # Find any image file in the data folder
        test_image_path = None
        for ext in image_extensions:
            pattern = os.path.join(data_folder, ext)
            matches = glob.glob(pattern, recursive=False)
            if matches:
                test_image_path = matches[0]  # Use the first image found
                break
        
        if not test_image_path:
            return {
                "products": [],
                "total_results": 0,
                "test_mode": True,
                "test_image_name": None,
                "recommendation_summary": "No test image found in data folder."
            }
        
        # Embed the test image
        emb = _image_embeddings.embed_image(test_image_path)
        results = _vectorstore_image.similarity_search_by_vector(emb, k=5)
        
        # Convert to structured product cards
        product_cards = docs_to_product_cards(results)
        test_image_name = os.path.basename(test_image_path)
        recommendation_summary = f"Found {len(product_cards)} visually similar products to {test_image_name}."
        
        return {
            "products": product_cards,
            "total_results": len(product_cards),
            "test_mode": True,
            "test_image_name": test_image_name,
            "recommendation_summary": recommendation_summary
        }
        
    except Exception as e:
        return {
            "products": [],
            "total_results": 0,
            "test_mode": True,
            "test_image_name": None,
            "recommendation_summary": f"Error processing image search: {str(e)}"
        }

if __name__ == "__main__":
    agent = main()
    print("Shopping agent initialized successfully!")
    # Example usage:
    # response = agent.invoke({"input": "What products do you recommend for outdoor activities?"})
    # print(response["output"])