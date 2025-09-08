#!/usr/bin/env python3
"""
Quick test of the two-phase image processing approach
"""

import os
import sys
from pathlib import Path

# add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def test_two_phase_approach():
    """Test the two-phase download → embed approach with a small dataset"""
    
    print(" Testing Two-Phase Image Processing")
    print("=" * 50)
    
    try:
        from app.vectordb_image_optimized import OptimizedImageVectorDBBuilder
        import torch
        
        # initialize builder
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"  Device: {device}")
        
        builder = OptimizedImageVectorDBBuilder(device=device)
        
        # load a small sample for testing
        print(" Loading sample data...")
        products_df, categories_df = builder.load_data()
        
        # test with just 50 products
        test_df = products_df.head(50)
        print(f" Testing with {len(test_df)} products")
        
        # test two-phase processing
        print("\n Starting Two-Phase Processing Test...")
        
        processed_images = builder.two_phase_process_images(
            test_df, 
            categories_df,
            download_workers=4,      # conservative for testing
            embedding_batch_size=16  # small batches for testing
        )
        
        if processed_images:
            print(f"\n Two-Phase Test Successful!")
            print(f"   Processed: {len(processed_images)} images")
            print(f"   Sample embedding shape: {len(processed_images[0]['embedding'])}")
            print(f"   Cache directory: image_cache/")
            
            # show some stats
            categories = set(img['metadata']['category'] for img in processed_images)
            bestsellers = sum(1 for img in processed_images if img['metadata']['isBestSeller'])
            
            print(f"\n Test Results:")
            print(f"   Categories represented: {len(categories)}")
            print(f"   Bestsellers included: {bestsellers}")
            print(f"   Average rating: {sum(img['metadata']['stars'] for img in processed_images) / len(processed_images):.2f}")
            
            print(f"\n Ready for full database build!")
            print(f"   Run: python build_image_database_fast.py")
            
        else:
            print(" Two-phase test failed - no images processed")
            
    except ImportError as e:
        print(f" Import error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f" Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_two_phase_approach()
