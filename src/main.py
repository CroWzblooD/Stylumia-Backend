# backend/src/main.py

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from .extractor.product_extractor import ProductExtractor
import os
import pandas as pd
import numpy as np
import traceback
import math
import asyncio
import threading
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from .recommendation.recommendation_engine import RecommendationEngine
import json
import io
import logging
from pydantic import BaseModel

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def clean_nan_values(obj):
    """Recursively clean NaN values from dictionary/list"""
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(x) for x in obj]
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj) if not np.isnan(obj) else None
    elif pd.isna(obj):
        return None
    return obj

# Global state for extraction progress
extraction_progress = {
    'total_records': {},
    'processed_records': {},
    'is_extracting': {},
    'cached_data': {}
}

@app.get("/api/extraction-progress/{category}")
async def get_extraction_progress(category: str):
    return {
        "total": extraction_progress['total_records'].get(category, 0),
        "processed": extraction_progress['processed_records'].get(category, 0),
        "is_extracting": extraction_progress['is_extracting'].get(category, False)
    }

# Global thread pool for processing
thread_pool = ThreadPoolExecutor(max_workers=4)

async def process_chunk(chunk: pd.DataFrame, extractor: ProductExtractor) -> List[Dict[Any, Any]]:
    """Process a chunk of products in parallel"""
    def extract_features(row):
        return extractor.extract_basic_features(row)
    
    # Process chunk in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    features = await loop.run_in_executor(
        thread_pool,
        lambda: [extract_features(row) for _, row in chunk.iterrows()]
    )
    return [f for f in features if f]

async def extract_and_cache_data(category: str, file_path: str):
    try:
        chunk_size = 100  # Process in smaller chunks
        
        # Count total records
        total_rows = sum(1 for _ in open(file_path, encoding='utf-8')) - 1
        extraction_progress['total_records'][category] = total_rows
        
        # Initialize cache if not exists
        if category not in extraction_progress['cached_data']:
            extraction_progress['cached_data'][category] = []
        
        # Process in chunks
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, encoding='utf-8'):
            products = []
            for _, row in chunk.iterrows():
                product = product_extractor.extract_basic_features(row)
                if product:
                    products.append(product)
            
            # Update cache with new products
            extraction_progress['cached_data'][category].extend(products)
            
            # Save to disk cache periodically
            if len(extraction_progress['cached_data'][category]) % 1000 == 0:
                save_cache_to_disk(category)
            
            await asyncio.sleep(0.01)  # Allow other requests to be processed
        
        # Final cache save
        save_cache_to_disk(category)
        
    except Exception as e:
        print(f"Error in extract_and_cache_data: {e}")
    finally:
        extraction_progress['is_extracting'][category] = False

def save_cache_to_disk(category):
    """Save extracted data to disk cache"""
    try:
        cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_file = os.path.join(cache_dir, f"{category}_cache.json")
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(extraction_progress['cached_data'][category], f)
    except Exception as e:
        print(f"Error saving cache: {e}")

def load_cache_from_disk(category):
    """Load cached data from disk"""
    try:
        cache_file = os.path.join(os.path.dirname(__file__), "cache", f"{category}_cache.json")
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading cache: {e}")
    return None

@app.get("/api/products/{category}")
async def get_products(category: str, background_tasks: BackgroundTasks = None):
    try:
        base_path = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "public", "Data")
        
        category_files = {
            "dresses": "Dresses Data Dump.csv",
            "earrings": "Earrings Data Dump.csv",
            "jeans": "Jeans Data Dump.csv",
            "sarees": "Saree Data Dump.csv",
            "shirts": "shirts_data_dump.csv",
            "sneakers": "Sneakers Data Dump.csv",
            "tshirts": "Tshirts Data Dump.csv"
        }
        
        if category not in category_files:
            raise HTTPException(status_code=404, detail="Category not found")
            
        file_path = os.path.join(base_path, category_files[category])
        
        # Check if we have cached data
        if category in extraction_progress['cached_data']:
            return {
                "products": extraction_progress['cached_data'][category],
                "extraction_progress": {
                    "is_cached": True,
                    "total": len(extraction_progress['cached_data'][category]),
                    "processed": len(extraction_progress['cached_data'][category])
                }
            }
        
        # If extraction is not in progress, start it
        if category not in extraction_progress['is_extracting'] or not extraction_progress['is_extracting'][category]:
            extraction_progress['is_extracting'][category] = True
            extraction_progress['cached_data'][category] = []
            background_tasks.add_task(extract_and_cache_data, category, file_path)
        
        # Return currently extracted products
        current_products = extraction_progress['cached_data'].get(category, [])
        
        return {
            "products": current_products,
            "extraction_progress": {
                "is_cached": False,
                "total": extraction_progress['total_records'].get(category, 0),
                "processed": len(current_products)
            }
        }
        
    except Exception as e:
        print(f"Error in get_products: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Cache management
category_cache = {}
category_loading = {}

def is_category_cached(category: str) -> bool:
    return category in category_cache

def get_cached_data(category: str, page: int, page_size: int) -> pd.DataFrame:
    if category not in category_cache:
        return pd.DataFrame()  # Return empty if not cached
    
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    return category_cache[category].iloc[start_idx:end_idx]

def get_total_rows(category: str) -> int:
    if category in category_cache:
        return len(category_cache[category])
    return 0

async def cache_category_data(category: str, file_path: str):
    if category in category_loading:
        return
    
    category_loading[category] = True
    try:
        # Load full dataset in background
        df = pd.read_csv(file_path, encoding='utf-8')
        category_cache[category] = df
    finally:
        category_loading[category] = False

@app.get("/api/insights/{category}")
async def get_insights(category: str):
    try:
        if not product_extractor.trend_data:
            return {"message": "No insights available"}
            
        category_insights = product_extractor.trend_data.get(category, {})
        return category_insights
        
    except Exception as e:
        print(f"Error getting insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/fashion-insights/all")
async def get_all_fashion_insights():
    try:
        if not product_extractor.trend_data:
            return {"message": "No insights available"}
            
        return {
            "knowledge_graph": {
                "trend_patterns": product_extractor.trend_data,
                "style_relationships": product_extractor._extract_style_relationships()
            }
        }
    except Exception as e:
        print(f"Error in all fashion insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Initialize recommendation engine
recommendation_engine = RecommendationEngine()

@app.on_event("startup")
async def startup_event():
    """Initialize the ProductExtractor and load cached data"""
    try:
        global product_extractor, recommendation_engine
        product_extractor = ProductExtractor()
        
        # Initialize recommendation engine with all category data
        all_csv_files = []
        base_path = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "public", "Data")
        category_files = {
            "dresses": "Dresses Data Dump.csv",
            "earrings": "Earrings Data Dump.csv",
            "jeans": "Jeans Data Dump.csv",
            "sarees": "Saree Data Dump.csv",
            "shirts": "shirts_data_dump.csv",
            "sneakers": "Sneakers Data Dump.csv",
            "tshirts": "Tshirts Data Dump.csv"
        }
        
        # Collect all CSV files
        for category, filename in category_files.items():
            file_path = os.path.join(base_path, filename)
            if os.path.exists(file_path):
                all_csv_files.append(file_path)
        
        # Load data into recommendation engine
        recommendation_engine.load_data(all_csv_files)
        
        # Load cached data for all categories
        for category in category_files.keys():
            cached_data = load_cache_from_disk(category)
            if cached_data:
                extraction_progress['cached_data'][category] = cached_data
                extraction_progress['total_records'][category] = len(cached_data)
                print(f"Loaded {len(cached_data)} cached products for {category}")
                
    except Exception as e:
        print(f"Error during startup: {e}")
        product_extractor = ProductExtractor()

@app.get("/api/products/search/{category}")
async def search_products(
    category: str, 
    query: str, 
    page: int = 1, 
    page_size: int = 200,
    filters: dict = None
):
    try:
        if category not in extraction_progress['cached_data']:
            raise HTTPException(status_code=404, detail="Category not found or not cached")
            
        # Perform semantic search
        search_results = product_extractor.semantic_search(query)
        
        # Apply filters if provided
        if filters:
            search_results = apply_filters(search_results, filters)
            
        # Paginate results
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_results = search_results[start_idx:end_idx]
        
        return {
            "products": paginated_results,
            "total": len(search_results),
            "page": page,
            "pages": math.ceil(len(search_results) / page_size)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def apply_filters(products: List[Dict], filters: Dict) -> List[Dict]:
    """Apply filters to product list"""
    filtered_products = products
    
    for key, value in filters.items():
        if not value:
            continue
            
        if key == 'price_range':
            filtered_products = [
                p for p in filtered_products 
                if value['min'] <= p['price'] <= value['max']
            ]
        elif key == 'colors':
            filtered_products = [
                p for p in filtered_products 
                if any(c[0].lower() in value for c in p.get('colors', []))
            ]
        elif key == 'materials':
            filtered_products = [
                p for p in filtered_products 
                if any(m.lower() in value for m in p.get('materials', []))
            ]
        elif key == 'trends':
            filtered_products = [
                p for p in filtered_products 
                if p['ai_insights']['trend_analysis']['trend_status'] in value
            ]
            
    return filtered_products

class ProductInput(BaseModel):
    product_id: str
    category_name: str
    brand: Optional[str] = ''
    style_attributes: Optional[str] = ''
    material_composition: Optional[str] = ''
    color: Optional[str] = ''
    mrp: Optional[float] = 0.0
    product_name: Optional[str] = ''
    image_url: Optional[str] = ''

class RecommendationRequest(BaseModel):
    liked_products: List[ProductInput]
    purchase_history: Optional[List[ProductInput]] = []

@app.post("/api/recommendations")
async def get_recommendations(request: RecommendationRequest):
    try:
        if not recommendation_engine:
            raise HTTPException(status_code=500, detail="Recommendation engine not initialized")

        # Debug logging
        print(f"Received request with {len(request.liked_products)} liked products")
        print(f"Purchase history: {len(request.purchase_history)} items")

        # Ensure recommendation engine has latest data
        await update_recommendation_engine()

        # Get recommendations using the engine
        recommendations = recommendation_engine.get_recommendations(
            user_id="temp_user",  # Temporary user ID
            liked_products=[product.dict() for product in request.liked_products],
            purchase_history=[product.dict() for product in request.purchase_history],
            n_recommendations=20
        )
        
        # Clean NaN values from recommendations
        cleaned_recommendations = [clean_nan_values(rec) for rec in recommendations]
        
        # Debug logging
        print(f"Generated {len(cleaned_recommendations)} recommendations")
        
        return {
            "status": "success",
            "recommendations": cleaned_recommendations,
            "total": len(cleaned_recommendations)
        }
        
    except Exception as e:
        print(f"Recommendation error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

async def update_recommendation_engine():
    """Update recommendation engine with latest data"""
    try:
        # Collect all current product data
        all_products = []
        for category_data in extraction_progress['cached_data'].values():
            all_products.extend(category_data)
        
        # Create temporary DataFrame for the engine
        df = pd.DataFrame(all_products)
        
        # Update the recommendation engine's data
        recommendation_engine.update_data(df)
        
    except Exception as e:
        print(f"Error updating recommendation engine: {e}")
        traceback.print_exc()

@app.post("/api/image-search")
async def image_search(
    image: UploadFile = File(...),
    category: str = Form(...)
):
    try:
        logger.info(f"Received image search request for category: {category}")
        
        # Read image content
        image_content = await image.read()
        image_bytes = io.BytesIO(image_content)
        
        # Get products for the specific category
        category_products = extraction_progress['cached_data'].get(category, [])
        if not category_products:
            logger.warning(f"No products found for category: {category}")
            return {"products": [], "total": 0}
        
        # Find similar products
        similar_products = product_extractor.find_similar_products(
            query_image=image_bytes,
            products=category_products,
            category=category
        )
        
        logger.info(f"Found {len(similar_products)} similar products")
        return {
            "products": similar_products,
            "total": len(similar_products)
        }
        
    except Exception as e:
        logger.error(f"Image search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))