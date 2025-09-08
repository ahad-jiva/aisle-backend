#!/usr/bin/env python3
"""
Test script for the new structured API responses
"""

import requests
import json

def test_structured_product_search():
    """Test the structured product search endpoint"""
    print("Testing Structured Product Search API")
    print("=" * 50)
    
    # test product search
    url = "http://localhost:8000/search/products"
    payload = {
        "query": "wireless bluetooth headphones"
    }
    
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"Product Search Successful!")
            print(f"Query: {data['query']}")
            print(f"Total Results: {data['total_results']}")
            print(f"Summary: {data['recommendation_summary']}")
            print()
            
            print("Product Cards:")
            for i, product in enumerate(data['products'], 1):
                print(f"\n  {i}. {product['title']}")
                print(f"     Price: ${product['price']}")
                print(f"     Rating: {product['rating']}")
                print(f"     Category: {product['category']}")
                print(f"     Tier: {product['tier_label']}")
                print(f"     Bestseller: {product['is_bestseller']}")
                print(f"     Sales: {product['sales_volume']}")
                print(f"     ID: {product['id']}")
                
                if product.get('description'):
                    print(f"     Description: {product['description'][:100]}...")
        else:
            print(f" Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print(" Connection error - make sure the API is running on localhost:8000")
        print("   Run: python run_api.py")
    except Exception as e:
        print(f" Error: {e}")

def test_structured_image_search():
    """Test the structured image search endpoint"""
    print("\n  Testing Structured Image Search API")
    print("=" * 50)
    
    # test image search
    url = "http://localhost:8000/search/image"
    
    try:
        response = requests.post(url)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f" Image Search Successful!")
            print(f" Total Results: {data['total_results']}")
            print(f" Test Mode: {data['test_mode']}")
            print(f"  Test Image: {data.get('test_image_name', 'None')}")
            print(f" Summary: {data['recommendation_summary']}")
            print()
            
            if data['products']:
                print("  Product Cards:")
                for i, product in enumerate(data['products'], 1):
                    print(f"\n  {i}. {product['title']}")
                    print(f"      Price: ${product['price']}")
                    print(f"      Rating: {product['rating']}")
                    print(f"       Category: {product['category']}")
                    print(f"      Tier: {product['tier_label']}")
                    print(f"      ID: {product['id']}")
            else:
                print("  No products found or no test image available")
                
        else:
            print(f" Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print(" Connection error - make sure the API is running on localhost:8000")
        print("   Run: python run_api.py")
    except Exception as e:
        print(f" Error: {e}")

def test_chat_endpoint():
    """Test that the chat endpoint still works for conversational responses"""
    print("\n Testing Chat Endpoint (Should Still Work)")
    print("=" * 50)
    
    url = "http://localhost:8000/chat"
    payload = {
        "message": "What are the best gaming laptops under $1000?",
        "session_id": "test_session"
    }
    
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f" Chat Successful!")
            print(f" Response: {data['response'][:300]}...")
            print(f" Session: {data.get('session_id', 'None')}")
        else:
            print(f" Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print(" Connection error - make sure the API is running on localhost:8000")
    except Exception as e:
        print(f" Error: {e}")

def show_example_frontend_usage():
    """Show example of how frontend can use the structured data"""
    print("\n Example Frontend Usage")
    print("=" * 50)
    
    example_card = {
        "id": "abc123def456",
        "title": "Sony WH-1000XM4 Wireless Noise Canceling Headphones",
        "price": 348.0,
        "rating": 4.7,
        "category": "Electronics",
        "image_url": "https://example.com/image.jpg",
        "product_url": "https://example.com/product",
        "is_bestseller": True,
        "sales_volume": 2500,
        "tier": "premium",
        "tier_label": "Premium Choice",
        "description": "Industry-leading noise canceling with Dual Noise Sensor technology..."
    }
    
    print(" React Component Example:")
    print("""
function ProductCard({ product }) {
  return (
    <div className="product-card">
      <img src={product.image_url} alt={product.title} />
      <h3>{product.title}</h3>
      <div className="price">${product.price}</div>
      <div className="rating">{''.repeat(Math.round(product.rating))} {product.rating}</div>
      <div className="tier">{product.tier_label}</div>
      {product.is_bestseller && <span className="bestseller"> Bestseller</span>}
      <p>{product.description}</p>
      <a href={product.product_url} target="_blank">View Product</a>
    </div>
  );
}
""")
    
    print(f" Sample Product Data:")
    print(json.dumps(example_card, indent=2))

if __name__ == "__main__":
    print(" Testing Structured API Responses for Frontend Cards")
    print("=" * 60)
    
    # run all tests
    test_structured_product_search()
    test_structured_image_search()
    test_chat_endpoint()
    show_example_frontend_usage()
    
    print("\n Testing Complete!")
    print("\n Next Steps:")
    print("   1. Use the structured /search/products endpoint for product cards")
    print("   2. Use the structured /search/image endpoint for image search cards")
    print("   3. Use the /chat endpoint for conversational interactions")
    print("   4. Each product has all the fields needed for rich card components!")
