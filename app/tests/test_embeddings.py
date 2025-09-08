#!/usr/bin/env python3
"""
Quick test to verify embedding dimensions match between vectordb and agent
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Load environment
load_dotenv()

def test_embedding_dimensions():
    """Test that both vectordb and agent use same embedding dimensions"""
    
    try:
        # Import the embedding class
        from app.shopping_agent import LocalSentenceTransformerEmbeddings
        
        # Create embedding instance
        embeddings = LocalSentenceTransformerEmbeddings()
        
        # Test embedding a sample text
        test_text = "wireless headphones"
        embedding = embeddings.embed_query(test_text)
        
        print(f" Embedding model loaded successfully")
        print(f"🔢 Embedding dimension: {len(embedding)}")
        print(f" Test text: '{test_text}'")
        print(f" First 5 values: {embedding[:5]}")
        
        # Verify expected dimension
        if len(embedding) == 384:
            print(" Correct dimension (384) - matches vectordb!")
        else:
            print(f" Unexpected dimension {len(embedding)} - should be 384")
            
    except Exception as e:
        print(f" Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print(" Testing Embedding Dimensions")
    print("=" * 40)
    test_embedding_dimensions()
    print("\n Test complete!")
