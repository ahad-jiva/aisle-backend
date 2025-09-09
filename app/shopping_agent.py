import os
import math
import hashlib
from dotenv import load_dotenv
import json

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
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
    """format documents with smart product information including bestseller status"""
    if not docs:
        return "No products found."
    
    formatted_products = []
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata
        
        # extract key information
        title = metadata.get('title', 'Unknown Product')
        stars = metadata.get('stars', 0)
        price = metadata.get('price', 0)
        category = metadata.get('category', 'Unknown Category')
        is_bestseller = metadata.get('isBestSeller', False)
        sales_volume = metadata.get('boughtInLastMonth', 0)
        
        # format price
        price_str = f"${price}" if price > 0 else "Price not available"
        
        # format rating with visual indicator
        rating_str = f"{stars} stars" if stars > 0 else "No rating"
        
        # bestseller and sales indicators
        status_indicators = []
        if is_bestseller:
            status_indicators.append("BESTSELLER")
        
        if sales_volume > 1000:
            status_indicators.append(f"{sales_volume:,} sold this month")
        elif sales_volume > 0:
            status_indicators.append(f"{sales_volume} sold this month")
        
        # product tier indication
        if is_bestseller:
            tier_info = "Premium Choice"
        else:
            tier_info = "Value Alternative"
        
        # create smart product description
        product_info = f"""Product {i}: {title}
{tier_info} - {' | '.join(status_indicators) if status_indicators else 'Standard Product'}
Rating: {rating_str}
Price: {price_str}
Category: {category}
Description: {doc.page_content}"""
        
        formatted_products.append(product_info)
    
    return "\n\n" + "\n\n".join(formatted_products)

def docs_to_product_cards(docs) -> List[Dict[str, Any]]:
    """convert documents to structured product card data for frontend"""
    if not docs:
        return []
    
    product_cards = []
    for doc in docs:
        metadata = doc.metadata
        
        # extract key information
        title = metadata.get('title', 'Unknown Product')
        stars = float(metadata.get('stars', 0))
        price = float(metadata.get('price', 0))
        category = metadata.get('category', 'Unknown Category')
        is_bestseller = bool(metadata.get('isBestSeller', False))
        sales_volume = int(metadata.get('boughtInLastMonth', 0))
        product_url = metadata.get('productURL', '')
        image_url = metadata.get('imgUrl', '')
        
        # generate unique id from title and url
        product_id = hashlib.md5(f"{title}{product_url}".encode()).hexdigest()[:12]
        
        # determine tier
        tier = "premium" if is_bestseller else "value"
        tier_label = "Premium Choice" if is_bestseller else "Value Alternative"
        
        # create product card
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
    """create a summary of the recommendations for the frontend"""
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
    calculate smart ranking score based on multiple factors
    
    factors:
    - star rating (0-5): 40% weight
    - sales volume (boughtInLastMonth): 30% weight  
    - bestseller status: 30% weight
    """
    metadata = doc.metadata
    
    # extract and normalize ratings (0-5 scale)
    try:
        stars = float(metadata.get('stars', 0))
        stars = max(0, min(5, stars))  # clamp to 0-5 range
    except (ValueError, TypeError):
        stars = 0
    
    # extract and normalize sales volume (log scale for large numbers)
    try:
        sales = int(metadata.get('boughtInLastMonth', 0))
        sales = max(0, sales)
        # use log scale to prevent extreme sales from dominating
        normalized_sales = math.log10(sales + 1) / math.log10(10001)  # normalize to 0-1
    except (ValueError, TypeError):
        normalized_sales = 0
    
    # bestseller bonus
    is_bestseller = metadata.get('isBestSeller', False)
    bestseller_bonus = 1.0 if is_bestseller else 0.0
    
    # weighted score calculation
    rating_score = (stars / 5.0) * 0.4  # 40% weight for ratings
    sales_score = normalized_sales * 0.3  # 30% weight for sales volume
    bestseller_score = bestseller_bonus * 0.3  # 30% weight for bestseller status
    
    total_score = rating_score + sales_score + bestseller_score
    
    return total_score

def two_tier_product_retrieval(vectorstore, query: str, k_bestsellers: int = 8, k_alternatives: int = 7, final_count: int = 5):
    """
    smart two-tier product retrieval system
    
    tier 1: bestsellers ranked by ratings and sales
    tier 2: value alternatives (non-bestsellers) with high ratings and lower prices
    
    args:
        vectorstore: the milvus vector store
        query: search query
        k_bestsellers: number of bestseller candidates to retrieve
        k_alternatives: number of alternative candidates to retrieve
        final_count: final number of products to return
    
    returns:
        list of recommended products with mix of bestsellers and value alternatives
    """
    
    # tier 1: get bestselling products
    try:
        bestseller_filter = 'isBestSeller == true'
        bestsellers = vectorstore.similarity_search(
            query, 
            k=k_bestsellers,
            expr=bestseller_filter
        )
    except Exception as e:
        print(f"Bestseller search failed, falling back to unfiltered: {e}")
        # fallback if filtering not supported
        all_candidates = vectorstore.similarity_search(query, k=k_bestsellers * 2)
        bestsellers = [doc for doc in all_candidates if doc.metadata.get('isBestSeller', False)][:k_bestsellers]
    
    # sort bestsellers by smart ranking
    bestsellers_ranked = sorted(bestsellers, key=smart_ranking_score, reverse=True)
    
    # calculate average price of top bestsellers for value threshold
    bestseller_prices = []
    for doc in bestsellers_ranked[:3]:  # use top 3 bestsellers for price benchmark
        try:
            price = float(doc.metadata.get('price', 0))
            if price > 0:
                bestseller_prices.append(price)
        except (ValueError, TypeError):
            continue
    
    avg_bestseller_price = sum(bestseller_prices) / len(bestseller_prices) if bestseller_prices else 1000
    price_threshold = avg_bestseller_price * 0.8  # value alternatives should be 20% cheaper
    
    # tier 2: get value alternatives (non-bestsellers with good ratings and lower prices)
    try:
        alternative_filter = 'isBestSeller == false'
        alternatives = vectorstore.similarity_search(
            query,
            k=k_alternatives,
            expr=alternative_filter
        )
    except Exception as e:
        print(f"Alternative search failed, falling back to unfiltered: {e}")
        # fallback if filtering not supported
        all_candidates = vectorstore.similarity_search(query, k=k_alternatives * 2)
        alternatives = [doc for doc in all_candidates if not doc.metadata.get('isBestSeller', False)][:k_alternatives]
    
    # filter alternatives by price and rating criteria
    quality_alternatives = []
    for doc in alternatives:
        try:
            price = float(doc.metadata.get('price', 0))
            stars = float(doc.metadata.get('stars', 0))
            
            # criteria for value alternatives:
            # - price below threshold or good rating (4+ stars)
            # - minimum 3.5 star rating
            if (price < price_threshold or stars >= 4.0) and stars >= 3.5:
                quality_alternatives.append(doc)
        except (ValueError, TypeError):
            # include if we can't parse price/rating (assume it might be good)
            quality_alternatives.append(doc)
    
    # sort alternatives by rating and sales
    alternatives_ranked = sorted(quality_alternatives, key=smart_ranking_score, reverse=True)
    
    # combine results: mix of bestsellers and alternatives
    # aim for 60% bestsellers, 40% alternatives in final results
    bestseller_count = min(len(bestsellers_ranked), max(1, int(final_count * 0.6)))
    alternative_count = final_count - bestseller_count
    
    final_recommendations = []
    final_recommendations.extend(bestsellers_ranked[:bestseller_count])
    final_recommendations.extend(alternatives_ranked[:alternative_count])
    
    # if we don't have enough, fill with remaining from either tier
    if len(final_recommendations) < final_count:
        remaining_needed = final_count - len(final_recommendations)
        remaining_bestsellers = bestsellers_ranked[bestseller_count:]
        remaining_alternatives = alternatives_ranked[alternative_count:]
        
        # combine and sort remaining by score
        remaining = remaining_bestsellers + remaining_alternatives
        remaining_sorted = sorted(remaining, key=smart_ranking_score, reverse=True)
        final_recommendations.extend(remaining_sorted[:remaining_needed])
    
    return final_recommendations[:final_count]

class LocalSentenceTransformerEmbeddings(Embeddings):
    """local sentence transformer embeddings - same as used in vectordb construction"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """initialize with the same model used to build the vector database"""
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """embed multiple documents"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """embed a single query"""
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()

class CLIPImageEmbeddings(Embeddings):
    """custom embedding class for clip image embeddings"""
    
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """embed text documents - not used for image embeddings"""
        raise NotImplementedError("This embedding class is for images only")
    
    def embed_query(self, text: str) -> List[float]:
        """embed text query - not used for image embeddings"""
        raise NotImplementedError("This embedding class is for images only")
    
    def embed_image(self, image_path: str) -> List[float]:
        """embed a single image and return the embedding"""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # normalize the features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            return image_features.numpy().flatten().tolist()

def initialize_agent():

    # load llm
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-pro", google_api_key=SecretStr(GEMINI_API_KEY) if GEMINI_API_KEY else None,
        temperature=0.7
    )

    # system prompt for the agent (simple string)
    system_prompt = """

You are AI.sle, an AI shopping agent acting as a friendly and knowledgeable online store employee. Your job is to help customers explore products, answer questions, and recommend items from the store catalog. You must always ground your answers in the product database and search results provided to you.
If asked about your identity, you should say that you are AI.sle, an AI shopping agent acting as a friendly and knowledgeable online store employee.
Speak in a conversational, helpful, and friendly tone, like a store clerk who wants the customer to find exactly what they need. Avoid being overly robotic or overly casual. Balance warmth with expertise and knowledge. Be concise, but provide enough detail for clarity. All queries and responses must be in English.
You have access to a product recommendation system and image search system. Use these tools as necessary according to the user's query.
    
Some important guidelines:
1. The recommended products will be automatically sent to and formatted by the frontend. This means that your response should NOT be a list of product suggestions, but rather a more high level response. For example, instead of listing the products that the user can already see, say something like <I found these products that you may be interested in.>
2. If the user asks you to search for products, you should use the ProductSearch tool. You should only use this tool if the user explicitly asks you to search for products. Use as specific of a search query as possible to get the best results.
3. If the user uploads an image, you should use the ImageSearch tool.
4. If the user's search request is vague or ambiguous, feel free to ask for more clarification before using the tools. It is better to ask for more information than to make assumptions. Remember, the worst possible thing you can do is recommend products that the user did not ask for.
5. After providing the user with the recommended products, you should ask some follow up questions to help the user narrow down their search. This is a great way to keep the conversation going and show that you are listening to the user.
6. Your response should be conversational and engaging. You can use emojis and Markdown formatting to make your response more engaging. If you use Markdown formatting, you must use the correct syntax.
7. DO NOT under any circumstance use language that sounds like a sales pitch or persuades the user to buy products. Your goal is to help the user find the best products for their needs, not to sell them products. You are not a salesperson.
8. NEVER provide medical advice if the user asks about medical products. You are not a doctor. If the user asks for suggestions for medical products, explictly mention that you are NOT a doctor and that the user should consult a medical professional for help.


"""
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

    # create simple product retrieval function
    def smart_retriever(query: str):
        """simple vector similarity product search"""
        docs = vectorstore_text.similarity_search(query, k=6)
        # convert retrieved docs to structured product cards for frontend
        product_cards = docs_to_product_cards(docs)
        recommendation_summary = create_recommendation_summary(product_cards, query)
        payload = {
            "products": product_cards,
            "total_results": len(product_cards),
            "query": query,
            "recommendation_summary": recommendation_summary
        }
        return json.dumps(payload)

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
        """Find products similar to the provided image using visual similarity.
        Returns a JSON string with products and a summary to be parsed by the API/frontend.
        """
        try:
            import os
            
            # validate image path exists
            if not os.path.exists(image_path):
                return json.dumps({
                    "products": [],
                    "total_results": 0,
                    "error": f"Image file not found at path: {image_path}"
                })
            
            print(f"Processing image: {image_path}")
            
            # embed the provided image
            emb = image_embeddings.embed_image(image_path)
            results = vectorstore_image.similarity_search_by_vector(emb, k=6)
            
            # convert to structured product cards json
            product_cards = docs_to_product_cards(results)
            test_image_name = os.path.basename(image_path)
            recommendation_summary = f"Found {len(product_cards)} visually similar products to {test_image_name}."

            return json.dumps({
                "products": product_cards,
                "total_results": len(product_cards),
                "query": test_image_name,
                "recommendation_summary": recommendation_summary
            })
            
        except Exception as e:
            return json.dumps({
                "products": [],
                "total_results": 0,
                "error": f"Error processing image: {str(e)}"
            })

    # define langchain tools for model to use
    tools = [
        Tool(
            name = "ProductSearch",
            func=lambda q: smart_retriever(q),
            description="Use this to find recommended products from a catalog when the user asks about items"
        ),
        Tool(
            name = "ImageSearch",
            func=lambda image_path: image_search_tool(image_path),
            description="Use this to find products visually similar to a provided image. Pass the full path to the image file. This searches the product catalog using visual similarity."
        )
    ]

    checkpointer = InMemorySaver()

    agent = create_react_agent(llm, tools=tools, prompt=system_prompt, checkpointer=checkpointer)

    return agent

# global variables for structured api access
_vectorstore_text = None
_vectorstore_image = None
_image_embeddings = None

def get_structured_product_recommendations(query: str) -> Dict[str, Any]:
    """
    get structured product recommendations for api endpoints
    returns json-serializable data for frontend cards
    """
    global _vectorstore_text
    
    if _vectorstore_text is None:
        # initialize if not already done
        embedding_model = LocalSentenceTransformerEmbeddings()
        _vectorstore_text = Milvus(
            embedding_function=embedding_model,
            collection_name="palona_text",
            connection_args={
                "uri": ZILLIZ_URI,
                "token": ZILLIZ_TOKEN
            }
        )
    
    # simple vector similarity search
    smart_docs = _vectorstore_text.similarity_search(query, k=8)
    
    # convert to structured product cards
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
    get structured image search results for api endpoints
    returns json-serializable data for frontend cards
    """
    global _vectorstore_image, _image_embeddings
    
    if _vectorstore_image is None or _image_embeddings is None:
        # initialize if not already done
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
        # look for test image in data folder
        import glob
        
        data_folder = "data"
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.webp"]
        
        # find any image file in the data folder
        test_image_path = None
        for ext in image_extensions:
            pattern = os.path.join(data_folder, ext)
            matches = glob.glob(pattern, recursive=False)
            if matches:
                test_image_path = matches[0]  # use the first image found
                break
        
        if not test_image_path:
            return {
                "products": [],
                "total_results": 0,
                "test_mode": True,
                "test_image_name": None,
                "recommendation_summary": "No test image found in data folder."
            }
        
        # embed the test image
        emb = _image_embeddings.embed_image(test_image_path)
        results = _vectorstore_image.similarity_search_by_vector(emb, k=5)
        
        # convert to structured product cards
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
    agent = initialize_agent()
    print("Shopping agent initialized successfully!")
    # example usage:
    # response = agent.invoke({"messages": [{"role": "user", "content": "what products do you recommend for outdoor activities?"}]})
    # print(response["messages"][-1].content)