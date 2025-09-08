#!/usr/bin/env python3
"""
Test script for image search functionality using local test image
"""

import os
import sys
from pathlib import Path
import glob

# add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def check_test_image():
    """Check if there's a test image in the data folder"""
    data_folder = "data"
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.webp"]
    
    print(f" Looking for test images in {data_folder}/ folder...")
    
    found_images = []
    for ext in image_extensions:
        pattern = os.path.join(data_folder, ext)
        matches = glob.glob(pattern, recursive=False)
        found_images.extend(matches)
    
    if found_images:
        print(" Found test image(s):")
        for img in found_images:
            print(f"   - {img}")
        return found_images[0]
    else:
        print(f" No test images found in {data_folder}/ folder")
        print("Please add an image file to test with:")
        print("   - Supported formats: jpg, jpeg, png, gif, bmp, webp")
        print("   - Example: data/test_headphones.jpg")
        return None

def test_image_search():
    """Test the image search functionality"""
    print(" Testing Image Search Functionality")
    print("=" * 50)
    
    # check for test image
    test_image = check_test_image()
    if not test_image:
        return
    
    try:
        # import the shopping agent
        from app.shopping_agent import main
        
        print("\nInitializing shopping agent...")
        agent = main()
        print(" Agent initialized successfully!")
        
        # test image search
        print(f"\n  Testing visual search with: {os.path.basename(test_image)}")
        print("-" * 30)
        
        result = agent.invoke({
            "input": "Find products similar to the test image using visual search"
        })
        
        response = result.get("output", "No response")
        print("\n Visual Search Results:")
        print("=" * 60)
        print(response)
        print("=" * 60)
        
        print("\n Image search test completed successfully!")
        
    except ImportError as e:
        print(f" Import error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f" Error during image search test: {e}")
        import traceback
        traceback.print_exc()

def test_via_api():
    """Test image search via the API"""
    print("\n Testing via API...")
    
    try:
        import requests
        
        # test the api endpoint
        response = requests.post("http://localhost:8000/search/image")
        
        if response.status_code == 200:
            data = response.json()
            print(" API test successful!")
            print(f"Results: {data.get('results', 'No results')}")
        else:
            print(f" API test failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("  API server not running. Start it with: python run_api.py")
    except ImportError:
        print("  requests not installed. Install with: pip install requests")
    except Exception as e:
        print(f" API test error: {e}")

if __name__ == "__main__":
    print("  Image Search Test Suite")
    print("=" * 40)
    
    test_image_search()
    test_via_api()
    
    print("\n Tips:")
    print("   - Add different test images to data/ folder to test various products")
    print("   - Try images of electronics, clothing, books, etc.")
    print("   - Make sure your image vector database is built first")
    print("   - Start API server with: python run_api.py")
