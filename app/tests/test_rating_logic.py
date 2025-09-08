#!/usr/bin/env python3
"""
Test script to verify the enhanced rating-based product recommendation logic
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Load environment
load_dotenv()

def test_enhanced_retrieval():
    """Test the enhanced retrieval logic"""
    
    try:
        # Import required modules
        from app.shopping_agent import enhanced_product_retrieval, LocalSentenceTransformerEmbeddings
        from langchain_milvus import Milvus
        import os
        
        print("🔗 Connecting to vector database...")
        
        # Setup vector store connection
        embedding_model = LocalSentenceTransformerEmbeddings()
        
        vectorstore = Milvus(
            embedding_function=embedding_model,
            collection_name="palona_text",
            connection_args={
                "uri": os.getenv("ZILLIZ_URI"),
                "token": os.getenv("ZILLIZ_TOKEN")
            }
        )
        
        print(" Connected successfully!")
        
        # Test enhanced retrieval
        test_query = "wireless headphones"
        print(f"\n Testing enhanced retrieval with query: '{test_query}'")
        
        # Get enhanced results (10 candidates -> top 5 by rating)
        enhanced_results = enhanced_product_retrieval(vectorstore, test_query, k_candidates=10, k_final=5)
        
        print(f" Retrieved {len(enhanced_results)} products (top-rated)")
        
        # Display results with ratings
        for i, doc in enumerate(enhanced_results, 1):
            metadata = doc.metadata
            title = metadata.get('title', 'Unknown')[:60] + "..."
            stars = metadata.get('stars', 0)
            price = metadata.get('price', 0)
            reviews = metadata.get('reviews', 0)
            
            print(f"\n{i}. {title}")
            print(f"    Rating: {stars} stars ({reviews} reviews)")
            print(f"    Price: ${price}")
        
        # Verify sorting by rating
        ratings = [doc.metadata.get('stars', 0) for doc in enhanced_results]
        is_sorted = all(ratings[i] >= ratings[i+1] for i in range(len(ratings)-1))
        
        if is_sorted:
            print(f"\n Results correctly sorted by rating (highest first)")
            print(f" Ratings: {ratings}")
        else:
            print(f"\n Results NOT properly sorted by rating")
            print(f" Ratings: {ratings}")
        
        print(f"\n Enhanced retrieval test completed!")
        
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def test_agent_integration():
    """Test the full agent with enhanced retrieval"""
    
    try:
        print(f"\n🤖 Testing full shopping agent...")
        
        from app.shopping_agent import main
        
        # Initialize agent
        agent = main()
        print(" Agent initialized successfully!")
        
        # Test query
        test_query = "Show me the best wireless headphones"
        print(f" Testing with query: '{test_query}'")
        
        result = agent.invoke({"input": test_query})
        response = result.get("output", "No response")
        
        print(f"\n Agent Response:")
        print("=" * 50)
        print(response)
        print("=" * 50)
        
        # Check if response mentions ratings
        if "star" in response.lower() or "rating" in response.lower() or "" in response:
            print("\n Response includes rating information!")
        else:
            print("\n  Response doesn't clearly mention ratings")
        
    except Exception as e:
        print(f" Agent test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print(" Testing Enhanced Rating-Based Product Recommendations")
    print("=" * 60)
    
    # Test 1: Enhanced retrieval logic
    test_enhanced_retrieval()
    
    # Test 2: Full agent integration  
    test_agent_integration()
    
    print(f"\n All tests completed!")
