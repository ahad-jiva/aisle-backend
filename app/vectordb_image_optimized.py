#!/usr/bin/env python3
"""
Optimized Image Vector Database Construction Script

This script provides multiple optimization strategies for building the image database:
1. Strategic sampling by category
2. Parallel processing for downloads  
3. Batch processing for embeddings
4. Quality filtering
5. Resume capability
"""

import os
import sys
import pandas as pd
import requests
from typing import List, Dict, Any
from tqdm import tqdm
import time
from dotenv import load_dotenv
from PIL import Image
import io
import torch
from transformers import CLIPProcessor, CLIPModel
import concurrent.futures
import threading
from collections import defaultdict
import pickle
import hashlib

# Vector database and embedding imports
from langchain_milvus import Milvus
from langchain_core.embeddings import Embeddings

# Load environment variables
load_dotenv()

class OptimizedCLIPImageEmbeddings(Embeddings):
    """Optimized CLIP-based image embeddings with batch processing"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "auto"):
        """Initialize CLIP model with device optimization"""
        print(f"Loading CLIP model: {model_name}")
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        print(f"Using device: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self._precomputed_embeddings = {}
        
        # Thread-local storage for requests sessions
        self._local = threading.local()
        
    def get_session(self):
        """Get thread-local requests session"""
        if not hasattr(self._local, 'session'):
            self._local.session = requests.Session()
            self._local.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
        return self._local.session
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Milvus compatibility method"""
        if hasattr(self, '_precomputed_embeddings') and self._precomputed_embeddings:
            embeddings = []
            for text in texts:
                if text in self._precomputed_embeddings:
                    embeddings.append(self._precomputed_embeddings[text])
                else:
                    embeddings.append([0.0] * 512)
            return embeddings
        else:
            return [[0.0] * 512 for _ in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Milvus compatibility method"""
        return [0.0] * 512
    
    def download_image(self, image_url: str) -> Image.Image:
        """Download image with optimized session reuse"""
        session = self.get_session()
        response = session.get(image_url, timeout=5)  # Reduced timeout
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        
        # Resize if too large (for memory efficiency)
        max_size = 384  # Smaller than before for speed
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        return image
    
    def embed_image_batch(self, image_urls: List[str]) -> List[List[float]]:
        """Process multiple images in a batch for efficiency"""
        embeddings = []
        valid_images = []
        
        # Download all images for this batch
        for url in image_urls:
            try:
                image = self.download_image(url)
                valid_images.append(image)
            except Exception as e:
                print(f"Failed to download {url}: {e}")
                valid_images.append(None)
        
        # Process valid images in batch
        if valid_images:
            # LFilter out None values and process in batch
            batch_images = [img for img in valid_images if img is not None]
            if batch_images:
                inputs = self.processor(images=batch_images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    image_features = self.model.get_image_features(**inputs)
                    # Normalize features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    batch_embeddings = image_features.cpu().numpy().tolist()
                
                # Map back to original order
                embedding_iter = iter(batch_embeddings)
                for img in valid_images:
                    if img is not None:
                        embeddings.append(next(embedding_iter))
                    else:
                        embeddings.append(None)
            else:
                embeddings = [None] * len(valid_images)
        
        return embeddings


class OptimizedImageVectorDBBuilder:
    """Optimized builder with multiple acceleration strategies"""
    
    def __init__(self, device: str = "auto"):
        self.zilliz_uri = os.getenv("ZILLIZ_URI")
        self.zilliz_token = os.getenv("ZILLIZ_TOKEN")
        
        if not self.zilliz_uri or not self.zilliz_token:
            raise ValueError("ZILLIZ_URI and ZILLIZ_TOKEN must be set in environment variables")
        
        self.embeddings = OptimizedCLIPImageEmbeddings(device=device)
        self.image_collection_name = "palona_image"
        
        print("Initialized Optimized Image Vector DB Builder")
        print(f"Device: {self.embeddings.device}")
        print(f"Collection: {self.image_collection_name}")
    
    def strategic_sample_by_category(self, products_df: pd.DataFrame, categories_df: pd.DataFrame, 
                                   total_samples: int = 50000, min_per_category: int = 100) -> pd.DataFrame:
        """
        Strategic sampling to ensure representation across all categories
        """
        print(f"Strategic sampling: targeting {total_samples} products with min {min_per_category} per category")
        
        # Create category lookup
        category_lookup = dict(zip(categories_df['id'], categories_df['category_name']))
        
        # Group products by category
        category_groups = products_df.groupby('category_id')
        
        sampled_products = []
        remaining_budget = total_samples
        categories_processed = 0
        
        # First pass: ensure minimum representation per category
        for category_id, group in category_groups:
            category_name = category_lookup.get(category_id, f"Category_{category_id}")
            
            # Filter for products with valid images and good ratings
            valid_group = group[
                (group['imgUrl'].notna()) & 
                (group['imgUrl'].str.startswith('http')) &
                (group['stars'] >= 3.0)  # Quality filter
            ]
            
            if len(valid_group) == 0:
                continue
                
            # Sample for this category
            sample_size = min(min_per_category, len(valid_group), remaining_budget)
            if sample_size > 0:
                # Prioritize bestsellers and high-rated products
                category_sample = valid_group.nlargest(sample_size, ['isBestSeller', 'stars', 'boughtInLastMonth'])
                sampled_products.append(category_sample)
                remaining_budget -= sample_size
                categories_processed += 1
                
            if remaining_budget <= 0:
                break
        
        # Second pass: distribute remaining budget proportionally
        if remaining_budget > 0:
            for category_id, group in category_groups:
                if remaining_budget <= 0:
                    break
                    
                valid_group = group[
                    (group['imgUrl'].notna()) & 
                    (group['imgUrl'].str.startswith('http')) &
                    (group['stars'] >= 3.0)
                ]
                
                already_sampled = sum(len(df[df['category_id'] == category_id]) for df in sampled_products)
                available = len(valid_group) - already_sampled
                
                if available > 0:
                    additional_sample = min(available, remaining_budget // categories_processed)
                    if additional_sample > 0:
                        # Skip already sampled products
                        remaining_products = valid_group.iloc[already_sampled:]
                        additional = remaining_products.head(additional_sample)
                        sampled_products.append(additional)
                        remaining_budget -= additional_sample
        
        # Combine all samples
        final_sample = pd.concat(sampled_products, ignore_index=True) if sampled_products else pd.DataFrame()
        
        print(f"Sampled {len(final_sample)} products across {categories_processed} categories")
        return final_sample
    
    def download_all_images(self, products_df: pd.DataFrame, cache_dir: str = "image_cache", 
                           max_workers: int = 8) -> Dict[str, str]:
        """
        Phase 1: Download ALL images first to maximize GPU efficiency in Phase 2
        Returns mapping of image_url -> local_file_path
        """
        print(f"Phase 1: Downloading {len(products_df)} images with {max_workers} workers")
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Prepare download tasks
        download_tasks = []
        for idx, row in products_df.iterrows():
            if pd.notna(row['imgUrl']) and row['imgUrl'].startswith('http'):
                # LCreate filename from hash of URL for uniqueness
                url_hash = hashlib.md5(row['imgUrl'].encode()).hexdigest()
                local_path = os.path.join(cache_dir, f"{url_hash}.jpg")
                download_tasks.append((row['imgUrl'], local_path))
        
        # Download images in parallel (pure I/O, no GPU)
        downloaded_files = {}
        failed_downloads = 0
        
        def download_single_image(url_and_path):
            url, local_path = url_and_path
            try:
                if os.path.exists(local_path):
                    return url, local_path  # Already downloaded
                
                session = self.embeddings.get_session()
                response = session.get(url, timeout=10)
                response.raise_for_status()
                
                # Download and save to disk
                image = Image.open(io.BytesIO(response.content)).convert("RGB")
                
                # Resize for memory efficiency
                max_size = 384
                if max(image.size) > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                image.save(local_path, "JPEG", quality=85)
                return url, local_path
                
            except Exception as e:
                return url, None
        
        # Parallel download
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(total=len(download_tasks), desc="Downloading images") as pbar:
                for url, local_path in executor.map(download_single_image, download_tasks):
                    if local_path:
                        downloaded_files[url] = local_path
                    else:
                        failed_downloads += 1
                    pbar.update(1)
        
        print(f"Downloaded {len(downloaded_files)} images, failed: {failed_downloads}")
        print(f"Images cached in: {cache_dir}")
        return downloaded_files

    def embed_all_cached_images(self, downloaded_files: Dict[str, str], products_df: pd.DataFrame, 
                               categories_df: pd.DataFrame, batch_size: int = 64) -> List[Dict[str, Any]]:
        """
        Phase 2: Batch embed ALL downloaded images for maximum GPU efficiency
        """
        print(f"Phase 2: GPU batch embedding {len(downloaded_files)} images (batch size: {batch_size})")
        
        category_lookup = dict(zip(categories_df['id'], categories_df['category_name']))
        
        # Prepare embedding data
        embedding_data = []
        for idx, row in products_df.iterrows():
            if row['imgUrl'] in downloaded_files:
                embedding_data.append({
                    'local_path': downloaded_files[row['imgUrl']], 
                    'row': row
                })
        
        processed_images = []
        failed_embeddings = 0
        
        # LProcess in large batches for maximum GPU efficiency
        total_batches = (len(embedding_data) + batch_size - 1) // batch_size
        
        print(f"GPU processing {total_batches} batches of {batch_size} images each...")
        
        with tqdm(total=len(embedding_data), desc="GPU embedding") as pbar:
            for batch_start in range(0, len(embedding_data), batch_size):
                batch_data = embedding_data[batch_start:batch_start + batch_size]
                
                try:
                    # Load all images for this batch
                    batch_images = []
                    batch_metadata = []
                    
                    for item in batch_data:
                        try:
                            image = Image.open(item['local_path']).convert("RGB")
                            batch_images.append(image)
                            batch_metadata.append(item['row'])
                        except Exception as e:
                            print(f"Failed to load {item['local_path']}: {e}")
                            failed_embeddings += 1
                            continue
                    
                    if not batch_images:
                        pbar.update(len(batch_data))
                        continue
                    
                    # LBatch process with GPU (maximum efficiency!)
                    inputs = self.embeddings.processor(images=batch_images, return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.embeddings.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        image_features = self.embeddings.model.get_image_features(**inputs)
                        # Normalize features
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        batch_embeddings = image_features.cpu().numpy().tolist()
                    
                    # Create metadata for successful embeddings
                    for embedding, row in zip(batch_embeddings, batch_metadata):
                        category_name = category_lookup.get(row['category_id'], 'Unknown Category')
                        
                        metadata = {
                            'title': str(row['title']),
                            'category': category_name,
                            'price': float(row['price']) if pd.notna(row['price']) else 0.0,
                            'stars': float(row['stars']) if pd.notna(row['stars']) else 0.0,
                            'isBestSeller': bool(row['isBestSeller']) if pd.notna(row['isBestSeller']) else False,
                            'productURL': str(row['productURL']) if pd.notna(row['productURL']) else '',
                            'imgUrl': row['imgUrl']
                        }
                        
                        processed_images.append({
                            'embedding': embedding,
                            'metadata': metadata
                        })
                    
                    pbar.update(len(batch_data))
                    
                except Exception as e:
                    print(f"GPU batch processing failed: {e}")
                    failed_embeddings += len(batch_data)
                    pbar.update(len(batch_data))
        
        print(f"Successfully embedded {len(processed_images)} images, failed: {failed_embeddings}")
        return processed_images

    def two_phase_process_images(self, products_df: pd.DataFrame, categories_df: pd.DataFrame, 
                                download_workers: int = 8, embedding_batch_size: int = 64) -> List[Dict[str, Any]]:
        """
        Two-phase processing: Download first, then batch embed for maximum GPU efficiency
        """
        print(f"Two-Phase Processing: Download-first approach for maximum GPU efficiency")
        print(f"   Phase 1: Download with {download_workers} workers")
        print(f"   Phase 2: GPU batch embedding (batch size: {embedding_batch_size})")
        
        # Phase 1: Download everything
        downloaded_files = self.download_all_images(products_df, max_workers=download_workers)
        
        if not downloaded_files:
            print("No images downloaded successfully")
            return []
        
        # Phase 2: Batch embed everything for maximum GPU efficiency
        processed_images = self.embed_all_cached_images(
            downloaded_files, products_df, categories_df, batch_size=embedding_batch_size
        )
        
        return processed_images
    
    def save_progress(self, processed_images: List[Dict[str, Any]], filename: str):
        """Save processed images to resume later"""
        with open(filename, 'wb') as f:
            pickle.dump(processed_images, f)
        print(f"Progress saved to {filename}")
    
    def load_progress(self, filename: str) -> List[Dict[str, Any]]:
        """Load previously processed images"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                processed_images = pickle.load(f)
            print(f"Loaded {len(processed_images)} previously processed images from {filename}")
            return processed_images
        return []
    
    def create_image_vector_database(self, processed_images: List[Dict[str, Any]], batch_size: int = 200):
        """Create vector database with larger batch sizes"""
        print(f"Creating vector database with {len(processed_images)} images...")
        
        try:
            vectorstore = Milvus(
                embedding_function=self.embeddings,
                collection_name=self.image_collection_name,
                connection_args={
                    "uri": self.zilliz_uri,
                    "token": self.zilliz_token
                },
                auto_id=True,
                drop_old=True
            )
            print(f"Connected to Milvus collection: {self.image_collection_name}")
        except Exception as e:
            print(f"Error connecting to Milvus: {e}")
            raise
        
        # Process in larger batches for efficiency
        total_batches = (len(processed_images) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(0, len(processed_images), batch_size), 
                              desc="Adding to vector database", total=total_batches):
            
            batch = processed_images[batch_idx:batch_idx + batch_size]
            
            try:
                embeddings_batch = [item['embedding'] for item in batch]
                metadatas_batch = [item['metadata'] for item in batch]
                dummy_texts = [f"Image: {meta['title']}" for meta in metadatas_batch]
                
                # Store embeddings temporarily
                self.embeddings._precomputed_embeddings = dict(zip(dummy_texts, embeddings_batch))
                
                # Add to vector database
                vectorstore.add_texts(
                    texts=dummy_texts,
                    metadatas=metadatas_batch
                )
                
                # Clear temporary storage
                self.embeddings._precomputed_embeddings = {}
                
            except Exception as e:
                print(f"Error adding batch {batch_idx}: {e}")
                continue
        
        print(f"Successfully created vector database!")
        return vectorstore

    def build_optimized_database(self, strategy: str = "strategic_sample", **kwargs):
        """
        Build database with different optimization strategies
        
        Strategies:
        - "strategic_sample": Sample by category for diversity
        - "quality_filter": Filter by ratings and bestsellers
        - "parallel_full": Full dataset with parallel processing
        """
        start_time = time.time()
        
        try:
            print(f"Building database with strategy: {strategy}")
            
            # Load data
            products_df, categories_df = self.load_data()
            
            # Apply strategy
            if strategy == "strategic_sample":
                total_samples = kwargs.get('total_samples', 50000)
                min_per_category = kwargs.get('min_per_category', 100)
                products_df = self.strategic_sample_by_category(
                    products_df, categories_df, total_samples, min_per_category
                )
            
            elif strategy == "quality_filter":
                min_rating = kwargs.get('min_rating', 4.0)
                products_df = products_df[
                    (products_df['stars'] >= min_rating) | 
                    (products_df['isBestSeller'] == True)
                ]
                print(f"Filtered to {len(products_df)} high-quality products")
            
            # Check for saved progress
            progress_file = f"image_progress_{strategy}.pkl"
            processed_images = self.load_progress(progress_file)
            
            if not processed_images:
                # Process images with two-phase approach for maximum GPU efficiency
                download_workers = kwargs.get('download_workers', 8)
                embedding_batch_size = kwargs.get('embedding_batch_size', 64)
                processed_images = self.two_phase_process_images(
                    products_df, categories_df, download_workers, embedding_batch_size
                )
                
                # Save progress
                self.save_progress(processed_images, progress_file)
            
            if processed_images:
                # Create vector database
                vectorstore = self.create_image_vector_database(processed_images)
                
                end_time = time.time()
                total_time = end_time - start_time
                
                print(f"\nOptimized image database construction completed!")
                print(f"Strategy: {strategy}")
                print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
                print(f"Images processed: {len(processed_images)}")
                print(f"Collection name: {self.image_collection_name}")
                
                return vectorstore
            else:
                print("No images were successfully processed!")
                return None
                
        except Exception as e:
            print(f"Error building optimized database: {e}")
            raise

    def load_data(self, data_dir: str = "data/"):
        """Load data with basic filtering"""
        print("Loading Amazon product data...")
        
        categories_path = os.path.join(data_dir, "amazon_categories.csv")
        categories_df = pd.read_csv(categories_path)
        
        products_path = os.path.join(data_dir, "amazon_products.csv")
        products_df = pd.read_csv(products_path)
        
        # Pre-filter for valid image URLs
        initial_count = len(products_df)
        products_df = products_df[
            products_df['imgUrl'].notna() & 
            (products_df['imgUrl'] != '') & 
            products_df['imgUrl'].str.startswith('http')
        ]
        
        print(f"Loaded {len(categories_df)} categories")
        print(f"Filtered {len(products_df)} products with valid images (from {initial_count} total)")
        
        return products_df, categories_df


def main():
    """Main execution with strategy selection"""
    print("Optimized Amazon Image Vector Database Construction")
    print("=" * 60)
    
    strategies = {
        "1": ("strategic_sample", "Strategic sampling by category (50K products, fast)"),
        "2": ("quality_filter", "High-quality products only (ratings 4+ or bestsellers)"),
        "3": ("parallel_full", "Full dataset with parallel processing (slow but complete)")
    }
    
    print("Available strategies:")
    for key, (strategy, description) in strategies.items():
        print(f"  {key}. {description}")
    
    choice = input("\nSelect strategy (1-3): ").strip()
    
    if choice not in strategies:
        print("Invalid choice, using strategic sampling")
        choice = "1"
    
    strategy_name, _ = strategies[choice]
    
    try:
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        builder = OptimizedImageVectorDBBuilder(device=device)
        
        # Strategy-specific parameters
        if strategy_name == "strategic_sample":
            builder.build_optimized_database(
                strategy=strategy_name,
                total_samples=50000,
                min_per_category=100,
                batch_size=16,
                max_workers=4
            )
        elif strategy_name == "quality_filter":
            builder.build_optimized_database(
                strategy=strategy_name,
                min_rating=4.0,
                batch_size=16,
                max_workers=4
            )
        else:  # parallel_full
            builder.build_optimized_database(
                strategy=strategy_name,
                batch_size=8,  # Smaller batches for full dataset
                max_workers=2  # Conservative for stability
            )
        
        print("\nOptimized image vector database is ready!")
        
    except Exception as e:
        print(f"Failed to build optimized database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
