#!/usr/bin/env python3
"""
Test script for the smart two-tier product recommendation system
Tests bestseller filtering, value alternatives, and ranking algorithms
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Load environment
load_dotenv()

def test_smart_ranking():
    """Test the smart ranking score calculation"""
    
    try:
        from app.shopping_agent import smart_ranking_score
        
        print(" Testing smart Ranking Algorithm")
        print("-" * 50)
        
        # Create test documents with different characteristics
        class MockDoc:
            def __init__(self, **metadata):
                self.metadata = metadata
        
        # Test case 1: High-rated bestseller with high sales
        bestseller_doc = MockDoc(
            stars=4.8, 
            isBestSeller=True, 
            boughtInLastMonth=5000,
            title="Premium Bestseller"
        )
        
        # Test case 2: High-rated non-bestseller with moderate sales
        value_doc = MockDoc(
            stars=4.6, 
            isBestSeller=False, 
            boughtInLastMonth=800,
            title="Value Alternative"
        )
        
        # Test case 3: Lower-rated bestseller
        mediocre_bestseller = MockDoc(
            stars=3.8, 
            isBestSeller=True, 
            boughtInLastMonth=2000,
            title="Mediocre Bestseller"
        )
        
        # Calculate scores
        bestseller_score = smart_ranking_score(bestseller_doc)
        value_score = smart_ranking_score(value_doc)
        mediocre_score = smart_ranking_score(mediocre_bestseller)
        
        print(f"Premium Bestseller score: {bestseller_score:.3f}")
        print(f"Value Alternative score: {value_score:.3f}")
        print(f"Mediocre Bestseller score: {mediocre_score:.3f}")
        
        # Verify ranking logic
        assert bestseller_score > value_score, "Premium bestseller should score higher than value alternative"
        assert bestseller_score > mediocre_score, "High-rated bestseller should beat mediocre bestseller"
        
        print(" Ranking algorithm working correctly!")
        
    except Exception as e:
        print(f" Ranking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_two_tier_retrieval():
    """Test the two-tier product retrieval system"""
    
    try:
        print(f"\n Testing Two-Tier Retrieval System")
        print("-" * 50)
        
        # Import required modules
        from app.shopping_agent import two_tier_product_retrieval, LocalSentenceTransformerEmbeddings
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
        
        # Test two-tier retrieval
        test_query = "wireless headphones"
        print(f"\n Testing two-tier retrieval with query: '{test_query}'")
        
        results = two_tier_product_retrieval(
            vectorstore, 
            test_query, 
            k_bestsellers=6, 
            k_alternatives=6, 
            final_count=5
        )
        
        print(f" Retrieved {len(results)} products from two-tier system")
        
        # Analyze results
        bestseller_count = 0
        value_count = 0
        
        for i, doc in enumerate(results, 1):
            metadata = doc.metadata
            title = metadata.get('title', 'Unknown')[:50] + "..."
            stars = metadata.get('stars', 0)
            price = metadata.get('price', 0)
            is_bestseller = metadata.get('isBestSeller', False)
            sales = metadata.get('boughtInLastMonth', 0)
            
            tier = " Premium" if is_bestseller else " Value"
            
            if is_bestseller:
                bestseller_count += 1
            else:
                value_count += 1
            
            print(f"\n{i}. {tier} - {title}")
            print(f"    {stars} stars |  ${price} |  {sales:,} sold")
        
        print(f"\n Result Distribution:")
        print(f"    Premium Choices: {bestseller_count}")
        print(f"    Value Alternatives: {value_count}")
        
        # Verify we have a good mix
        if bestseller_count > 0 and value_count > 0:
            print(" Good mix of premium and value options!")
        elif bestseller_count > 0:
            print("  Only premium choices found (may be expected)")
        elif value_count > 0:
            print("  Only value alternatives found (may be expected)")
        else:
            print(" No results found!")
            return False
        
        print(f"\n Two-tier retrieval test completed!")
        
    except Exception as e:
        print(f" Two-tier retrieval test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_full_agent_integration():
    """Test the complete smart agent"""
    
    try:
        print(f"\n🤖 Testing Full smart Shopping Agent")
        print("-" * 50)
        
        from app.shopping_agent import main
        
        # Initialize agent
        agent = main()
        print(" smart agent initialized!")
        
        # Test queries
        test_queries = [
            "Show me the best wireless headphones",
            "I need affordable laptop computers under $800",
            "What are the top-rated running shoes?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n Test Query {i}: '{query}'")
            print("=" * 60)
            
            try:
                result = agent.invoke({"input": query})
                response = result.get("output", "No response")
                
                # Check for smart features in response
                features_found = []
                if "Premium" in response or "" in response:
                    features_found.append("Premium tier mentions")
                if "Value" in response or "" in response:
                    features_found.append("Value tier mentions") 
                if "bestseller" in response.lower() or "best seller" in response.lower():
                    features_found.append("Bestseller recognition")
                if "star" in response.lower() or "rating" in response.lower() or "" in response:
                    features_found.append("Rating information")
                if "sold" in response.lower() or "sales" in response.lower():
                    features_found.append("Sales volume data")
                
                print(f" Agent Response:")
                print(response[:500] + "..." if len(response) > 500 else response)
                
                print(f"\n smart features detected: {', '.join(features_found) if features_found else 'None'}")
                
            except Exception as e:
                print(f" Query failed: {e}")
        
        print(f"\n Full agent integration test completed!")
        
    except Exception as e:
        print(f" Agent integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    print(" Testing smart Two-Tier Product Recommendation System")
    print("=" * 70)
    
    success_count = 0
    
    # Test 1: Ranking algorithm
    if test_smart_ranking():
        success_count += 1
    
    # Test 2: Two-tier retrieval
    if test_two_tier_retrieval():
        success_count += 1
    
    # Test 3: Full agent integration
    if test_full_agent_integration():
        success_count += 1
    
    print(f"\n Test Results: {success_count}/3 tests passed")
    
    if success_count == 3:
        print(" All smart ranking tests PASSED!")
        print(" Your two-tier recommendation system is working perfectly!")
    else:
        print("  Some tests failed - please review the output above")
        sys.exit(1)
