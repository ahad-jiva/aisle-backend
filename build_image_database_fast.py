#!/usr/bin/env python3
"""
Fast Image Database Builder with Multiple Optimization Strategies
"""

import os
import sys
from pathlib import Path

# add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def main():
    print("Fast Amazon Image Vector Database Builder")
    print("=" * 50)
    
    # check if data exists
    if not os.path.exists("data/amazon_products.csv"):
        print("Missing data/amazon_products.csv")
        sys.exit(1)
    
    print("Two-Phase Optimization Strategies (Download -> GPU Batch Embed):")
    print()
    print("1. Strategic Sampling (RECOMMENDED)")
    print("   • 25K products across all categories")
    print("   • Download first, then GPU batch embed")
    print("   • Maximum GPU efficiency, ~10-20 minutes")
    print()
    print("2. Quality Filter")
    print("   • Only 4+ star products and bestsellers")
    print("   • Two-phase: I/O then pure GPU processing")
    print("   • ~100K-200K products, ~30-60 minutes")
    print()
    print("3. Speed Test (100 products)")
    print("   • Quick test of two-phase pipeline")
    print("   • ~1-3 minutes")
    print()
    print("4. Custom Configuration")
    print("   • Tune download workers & GPU batch size")
    print()
    
    choice = input("Select strategy (1-4): ").strip()
    
    try:
        from app.vectordb_image_optimized import OptimizedImageVectorDBBuilder
        import torch
        
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            
        print(f"Using device: {device}")
        
        builder = OptimizedImageVectorDBBuilder(device=device)
        
        if choice == "1":
            print("\nBuilding with Strategic Sampling (Two-Phase)...")
            builder.build_optimized_database(
                strategy="strategic_sample",
                total_samples=25000,
                min_per_category=100,
                download_workers=8,
                embedding_batch_size=64
            )
            
        elif choice == "2":
            print("\nBuilding with Quality Filter (Two-Phase)...")
            builder.build_optimized_database(
                strategy="quality_filter",
                min_rating=4.0,
                download_workers=8,
                embedding_batch_size=64
            )
            
        elif choice == "3":
            print("\nRunning Speed Test (Two-Phase)...")
            # load and limit data
            products_df, categories_df = builder.load_data()
            products_df = products_df.head(100)
            
            processed_images = builder.two_phase_process_images(
                products_df, categories_df, download_workers=4, embedding_batch_size=32
            )
            
            if processed_images:
                builder.create_image_vector_database(processed_images)
                print("Speed test completed successfully!")
            else:
                print("Speed test failed")
                
        elif choice == "4":
            print("\nCustom Configuration (Two-Phase):")
            total_samples = int(input("Total samples (25000): ") or "25000")
            min_per_category = int(input("Min per category (100): ") or "100")
            download_workers = int(input("Download workers (8): ") or "8")
            embedding_batch_size = int(input("GPU batch size (64): ") or "64")
            
            builder.build_optimized_database(
                strategy="strategic_sample",
                total_samples=total_samples,
                min_per_category=min_per_category,
                download_workers=download_workers,
                embedding_batch_size=embedding_batch_size
            )
            
        else:
            print("Invalid choice, using strategic sampling")
            builder.build_optimized_database(strategy="strategic_sample")
        
        print("\nFast image database construction completed!")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
