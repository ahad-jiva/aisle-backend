#!/usr/bin/env python3
"""
Quick script to build the vector database
"""

import os
import sys
from pathlib import Path

# add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def main():
    print("  Amazon Vector Database Builder")
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
    
    # import and run the vector database builder
    try:
        from app.vectordb_text import AmazonVectorDBBuilder
        builder = AmazonVectorDBBuilder()
        builder.build_database()
        
    except ImportError as e:
        print(f" Import error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f" Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
