#!/usr/bin/env python3
"""
Quick script to build the image vector database
"""

import os
import sys
from pathlib import Path

# add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def main():
    print("  Amazon Image Vector Database Builder")
    print("=" * 40)
    
    # check if data exists
    if not os.path.exists("data/amazon_products.csv"):
        print(" Missing data/amazon_products.csv")
        print("Please ensure your Amazon dataset is in the data/ folder")
        sys.exit(1)
    
    if not os.path.exists("data/amazon_categories.csv"):
        print(" Missing data/amazon_categories.csv") 
        print("Please ensure your Amazon dataset is in the data/ folder")
        sys.exit(1)
    
    print("  WARNING: Image processing will download images from URLs")
    print("   This may take significantly longer than text processing")
    print("   Consider starting with a limited dataset for testing")
    
    response = input("\nProceed with image database construction? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Cancelled by user")
        sys.exit(0)
    
    # import and run the image vector database builder
    try:
        from app.vectordb_image import AmazonImageVectorDBBuilder
        builder = AmazonImageVectorDBBuilder()
        
        # ask about limiting dataset for testing
        limit_response = input("Limit to first 100 products for testing? (y/N): ")
        if limit_response.lower() in ['y', 'yes']:
            print("Building with limited dataset (100 products)...")
            builder.build_image_database(max_products=100)
        else:
            print("Building with full dataset...")
            builder.build_image_database()
        
    except ImportError as e:
        print(f" Import error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f" Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
