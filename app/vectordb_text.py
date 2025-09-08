#!/usr/bin/env python3
"""
Vector Database Construction Script for Amazon Products

This script processes 1.4M Amazon products from CSV files and creates a vector database
using Zilliz/Milvus for efficient product search and recommendations.
"""

import os
import sys
import pandas as pd
from typing import List, Dict, Any
from tqdm import tqdm
import time
from dotenv import load_dotenv

# vector database and embedding imports
from langchain_milvus import Milvus
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

# load environment variables
load_dotenv()

class LocalSentenceTransformerEmbeddings(Embeddings):
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()


class AmazonVectorDBBuilder:
    
    def __init__(self):
        self.zilliz_uri = os.getenv("ZILLIZ_URI")
        self.zilliz_token = os.getenv("ZILLIZ_TOKEN")
        
        if not self.zilliz_uri or not self.zilliz_token:
            raise ValueError("ZILLIZ_URI and ZILLIZ_TOKEN must be set in environment variables")
        
        # initialize embedding model
        self.embeddings = LocalSentenceTransformerEmbeddings()
        
        # collection names
        self.text_collection_name = "palona_text"
        self.image_collection_name = "palona_image"
        
        print("Initialized Vector DB Builder")
        print(f"Zilliz URI: {self.zilliz_uri}")
        print(f"Text Collection: {self.text_collection_name}")
    
    def load_data(self, data_dir: str = "data/") -> tuple[pd.DataFrame, pd.DataFrame]:
        print("Loading Amazon product data...")
        
        # load categories
        categories_path = os.path.join(data_dir, "amazon_categories.csv")
        categories_df = pd.read_csv(categories_path)
        print(f"Loaded {len(categories_df)} categories")
        
        # load products in chunks for memory efficiency
        products_path = os.path.join(data_dir, "amazon_products.csv")
        print(f"Loading products from {products_path}...")
        
        # read products csv - handle large file
        try:
            products_df = pd.read_csv(products_path)
            print(f"Loaded {len(products_df)} products")
        except Exception as e:
            print(f"Error loading products: {e}")
            # try reading in chunks if file is too large
            print("Attempting to read in chunks...")
            chunks = []
            chunk_size = 10000
            for chunk in pd.read_csv(products_path, chunksize=chunk_size):
                chunks.append(chunk)
                if len(chunks) % 10 == 0:
                    print(f"Loaded {len(chunks) * chunk_size} products...")
            products_df = pd.concat(chunks, ignore_index=True)
            print(f"Successfully loaded {len(products_df)} products in chunks")
        
        return products_df, categories_df
    
    def process_products(self, products_df: pd.DataFrame, categories_df: pd.DataFrame) -> List[Dict[str, Any]]:
        print("Processing product data...")
        
        # create category lookup
        category_lookup = dict(zip(categories_df['id'], categories_df['category_name']))
        
        processed_products = []
        
        for idx, row in tqdm(products_df.iterrows(), total=len(products_df), desc="Processing products"):
            try:
                # get category name
                category_name = category_lookup.get(row['category_id'], 'Unknown Category')
                
                # create rich text for embedding (title + category + price info)
                price_text = f"${row['price']}" if pd.notna(row['price']) and row['price'] > 0 else "Price not available"
                
                # combine information for better search
                combined_text = f"{row['title']} | Category: {category_name} | Price: {price_text}"
                
                # Add rating info if available
                if pd.notna(row['stars']) and row['stars'] > 0:
                    combined_text += f" | Rating: {row['stars']} stars"
                
                # Create optimized metadata for retrieval 
                # Only essential fields for recommendations + minimal frontend support
                metadata = {
                    # Core recommendation fields (always needed)
                    'title': str(row['title']),
                    'category': category_name,
                    'price': float(row['price']) if pd.notna(row['price']) else 0.0,
                    'stars': float(row['stars']) if pd.notna(row['stars']) else 0.0,
                    'isBestSeller': bool(row['isBestSeller']) if pd.notna(row['isBestSeller']) else False,
                    'boughtInLastMonth': int(row['boughtInLastMonth']) if pd.notna(row['boughtInLastMonth']) else 0,
                    
                    # Frontend support (minimal set)
                    'productURL': str(row['productURL']) if pd.notna(row['productURL']) else '',
                    'imgUrl': str(row['imgUrl']) if pd.notna(row['imgUrl']) else ''
                    
                    # Removed fields:
                    # - 'category_id': Redundant (already resolved to category name)
                    # - 'listPrice': Unused (no discount comparison logic currently)
                    # - 'reviews': All values are 0 (no discriminative value)
                }
                
                processed_products.append({
                    'text': combined_text,
                    'metadata': metadata
                })
                
            except Exception as e:
                print(f"Error processing product at index {idx}: {e}")
                continue
        
        print(f"Successfully processed {len(processed_products)} products")
        return processed_products
    
    def create_vector_database(self, processed_products: List[Dict[str, Any]], batch_size: int = 1000):
        print(f"Creating vector database with {len(processed_products)} products...")
        
        # initialize milvus connection
        try:
            vectorstore = Milvus(
                embedding_function=self.embeddings,
                collection_name=self.text_collection_name,
                connection_args={
                    "uri": self.zilliz_uri,
                    "token": self.zilliz_token
                },
                index_params={
                    "index_type": "HNSW",
                    "metric_type": "COSINE"
                },
                auto_id=True,
                drop_old=True  # This will recreate the collection
            )
            print(f"Connected to Milvus collection: {self.text_collection_name}")
        except Exception as e:
            print(f"Error connecting to Milvus: {e}")
            raise
        
        # process in batches for memory efficiency
        total_batches = (len(processed_products) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(0, len(processed_products), batch_size), 
                              desc="Adding to vector database", total=total_batches):
            
            batch = processed_products[batch_idx:batch_idx + batch_size]
            
            try:
                # extract texts and metadata for this batch
                texts = [item['text'] for item in batch]
                metadatas = [item['metadata'] for item in batch]
                
                # add to vector database
                vectorstore.add_texts(
                    texts=texts,
                    metadatas=metadatas
                )
                
                if batch_idx % (batch_size * 10) == 0:  # progress update every 10 batches
                    print(f"Processed {batch_idx + len(batch)} / {len(processed_products)} products")
                
            except Exception as e:
                print(f"Error adding batch {batch_idx}: {e}")
                continue
        
        print(f"Successfully created vector database with {len(processed_products)} products!")
        return vectorstore
    
    def build_database(self, data_dir: str = "data/"):
        # main method to build the complete vector database
        start_time = time.time()
        
        try:
            # load data
            products_df, categories_df = self.load_data(data_dir)
            
            # process products
            processed_products = self.process_products(products_df, categories_df)
            
            # create vector database
            vectorstore = self.create_vector_database(processed_products)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print("\nVector database construction completed!")
            print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            print(f"Products processed: {len(processed_products)}")
            print(f"Collection name: {self.text_collection_name}")
            print(f"Zilliz URI: {self.zilliz_uri}")
            
            # test the database
            self.test_database(vectorstore)
            
        except Exception as e:
            print(f"Error building database: {e}")
            raise
    
    def test_database(self, vectorstore):
        # test the vector database with a sample query
        print("\nTesting vector database...")
        
        try:
            # test query
            test_query = "wireless headphones"
            results = vectorstore.similarity_search(test_query, k=3)
            
            print(f"Test query: '{test_query}'")
            print(f"Results found: {len(results)}")
            
            for i, result in enumerate(results, 1):
                metadata = result.metadata
                print(f"\n{i}. {metadata.get('title', 'No title')[:100]}...")
                print(f"   Category: {metadata.get('category', 'Unknown')}")
                print(f"   Price: ${metadata.get('price', 0)}")
                print(f"   Rating: {metadata.get('stars', 0)} stars")
            
            print("\nDatabase test successful!")
            
        except Exception as e:
            print(f"Database test failed: {e}")


def main():
    # main execution function
    print("Starting Amazon Vector Database Construction")
    print("=" * 60)
    
    # check if data directory exists
    data_dir = "data/"
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found!")
        print("Please ensure the Amazon CSV files are in the data/ folder:")
        print("  - data/amazon_products.csv")
        print("  - data/amazon_categories.csv")
        sys.exit(1)
    
    # initialize and build database
    try:
        builder = AmazonVectorDBBuilder()
        builder.build_database(data_dir)
        
        print("\nVector database is ready for use!")
        print("You can now run your shopping agent with: python run_api.py")
        
    except Exception as e:
        print(f"Failed to build vector database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
