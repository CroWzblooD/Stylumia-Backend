from functools import lru_cache
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import traceback

class RecommendationEngine:
    def __init__(self):
        self.product_data = {}
        self.tfidf_matrices = {}
        self.vectorizers = {}
        self.current_likes = set()  # Track current liked product IDs
        self.current_categories = set()  # Track current liked categories
        self.category_features = {
            'dresses': ['dress', 'gown', 'frock', 'denim dress'],
            'tshirts': ['tshirt', 't-shirt', 'top'],
            'jeans': ['jeans', 'denim', 'pants'],
            'sarees': ['saree', 'sari'],
            'shirts': ['shirt', 'formal shirt'],
            'sneakers': ['shoes', 'sneakers', 'footwear'],
            'earrings': ['earring', 'jewelry']
        }
        # Preload all data at initialization
        self.load_all_data()
        print("Recommendation engine initialized with preloaded data")

    def load_all_data(self):
        """Preload all category data at initialization"""
        try:
            print("Preloading all category data...")
            base_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "frontend", "public", "Data")
            category_files = {
                "dresses": "Dresses Data Dump.csv",
                "earrings": "Earrings Data Dump.csv",
                "jeans": "Jeans Data Dump.csv",
                "sarees": "Saree Data Dump.csv",
                "shirts": "shirts_data_dump.csv",
                "sneakers": "Sneakers Data Dump.csv",
                "tshirts": "Tshirts Data Dump.csv"
            }
            
            for category, filename in category_files.items():
                file_path = os.path.join(base_path, filename)
                try:
                    if os.path.exists(file_path):
                        df = pd.read_csv(file_path)
                        df['category_name'] = category
                        self.product_data[category] = df
                        self._prepare_features(category)
                        print(f"Loaded {len(df)} products for {category}")
                    else:
                        print(f"Warning: File not found - {file_path}")
                except Exception as e:
                    print(f"Error loading {category} data: {e}")
                    continue
                    
            print("All data preloaded successfully")
            print(f"Available categories: {list(self.product_data.keys())}")
            
        except Exception as e:
            print(f"Error in load_all_data: {e}")
            traceback.print_exc()

    def update_likes(self, liked_products):
        """Update current likes and their categories"""
        try:
            new_likes = set()
            new_categories = set()
            
            for product in liked_products:
                product_id = product.get('product_id')
                if product_id:
                    new_likes.add(product_id)
                    category = self._map_category(
                        category_name=product.get('category_name'),
                        product_id=product_id,
                        product_name=product.get('product_name')
                    )
                    if category:
                        new_categories.add(category)
            
            # Check what changed
            added_likes = new_likes - self.current_likes
            removed_likes = self.current_likes - new_likes
            
            if added_likes:
                print(f"New likes added: {len(added_likes)} products")
            if removed_likes:
                print(f"Likes removed: {len(removed_likes)} products")
                
            # Update current state
            self.current_likes = new_likes
            self.current_categories = new_categories
            
            print(f"Current categories: {self.current_categories}")
            return True
            
        except Exception as e:
            print(f"Error updating likes: {e}")
            return False

    def get_recommendations(self, user_id, liked_products, purchase_history, n_recommendations=20):
        """Get recommendations based on current likes only"""
        try:
            # Update current likes first
            self.update_likes(liked_products)
            
            if not self.current_likes:
                print("No current likes, returning trending items")
                return self._get_trending_items(n_recommendations)
            
            # Get recommendations only from current categories
            all_recommendations = []
            items_per_category = max(5, n_recommendations // len(self.current_categories)) if self.current_categories else 0
            
            for category in self.current_categories:
                print(f"Getting recommendations for {category}")
                category_recs = self._get_category_recommendations(
                    category, 
                    tuple(sorted(self.current_likes)),  # Sort for consistent caching
                    items_per_category
                )
                all_recommendations.extend(category_recs)
            
            # Sort and balance recommendations
            all_recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
            final_recommendations = []
            
            # Ensure balanced representation from each category
            remaining_slots = n_recommendations
            while remaining_slots > 0 and all_recommendations:
                for category in self.current_categories:
                    category_items = [r for r in all_recommendations 
                                   if r['category_name'] == category]
                    if category_items:
                        final_recommendations.append(category_items.pop(0))
                        all_recommendations.remove(final_recommendations[-1])
                        remaining_slots -= 1
                        if remaining_slots <= 0:
                            break
            
            # Print summary
            rec_summary = {}
            for rec in final_recommendations:
                cat = rec['category_name']
                if cat not in rec_summary:
                    rec_summary[cat] = 0
                rec_summary[cat] += 1
            
            print("\nRecommendation Summary:")
            for cat, count in rec_summary.items():
                print(f"{cat}: {count} items")
            
            return final_recommendations
            
        except Exception as e:
            print(f"Error in get_recommendations: {e}")
            return []

    def update_data(self, new_data: pd.DataFrame):
        """Handle data updates without extraction"""
        return True  # Data is already preloaded 

    def _map_category(self, category_name=None, product_id=None, product_name=None):
        """Map product to a standard category"""
        try:
            # Try mapping from category_name first
            if category_name:
                category_name = str(category_name).lower().strip()
                # Direct match
                if category_name in self.category_features:
                    return category_name
                # Check features
                for main_category, features in self.category_features.items():
                    if category_name in features:
                        return main_category
                    for feature in features:
                        if feature in category_name:
                            return main_category

            # Try mapping from product_id
            if product_id:
                product_id = str(product_id).lower()
                for category in self.category_features.keys():
                    if category in product_id:
                        return category

            # Try mapping from product_name
            if product_name:
                product_name = str(product_name).lower()
                for category, features in self.category_features.items():
                    for feature in features:
                        if feature in product_name:
                            return category

            return None
            
        except Exception as e:
            print(f"Error in _map_category: {e}")
            return None

    def _get_category_recommendations(self, category, liked_product_ids, n_items):
        """Get recommendations for a specific category"""
        try:
            df = self.product_data.get(category)
            if df is None:
                print(f"No data found for category: {category}")
                return []
            
            # Get product data from IDs
            liked_products = []
            for product_id in liked_product_ids:
                product_data = df[df['product_id'] == product_id]
                if not product_data.empty:
                    liked_products.append(product_data.iloc[0])
            
            if not liked_products:
                print(f"No liked products found in category: {category}")
                return []
            
            # Ensure TF-IDF matrix exists
            if category not in self.tfidf_matrices:
                self._prepare_features(category)
            
            tfidf_matrix = self.tfidf_matrices.get(category)
            if tfidf_matrix is None:
                print(f"No TF-IDF matrix for category: {category}")
                return []
            
            # Calculate similarity scores
            similarity_scores = np.zeros(len(df))
            for product in liked_products:
                product_idx = df[df['product_id'] == product['product_id']].index
                if len(product_idx) > 0:
                    idx = product_idx[0]
                    similarities = cosine_similarity(
                        tfidf_matrix[idx:idx+1],
                        tfidf_matrix
                    ).flatten()
                    similarity_scores += similarities
            
            # Get recommendations
            already_liked = set(liked_product_ids)
            recommendations = []
            sorted_indices = similarity_scores.argsort()[::-1]
            
            for idx in sorted_indices:
                if len(recommendations) >= n_items:
                    break
                
                product = df.iloc[idx]
                if str(product['product_id']) not in already_liked:
                    recommendations.append({
                        'product_id': str(product['product_id']),
                        'product_name': str(product.get('product_name', '')),
                        'brand': str(product.get('brand', 'Generic')),
                        'category_name': category,
                        'mrp': float(product.get('mrp', 0.0)),
                        'feature_image': str(product.get('feature_image', '')),
                        'description': str(product.get('product_details', '')),
                        'material_composition': str(product.get('material_composition', '')),
                        'style_attributes': str(product.get('style_attributes', '')),
                        'dominant_color': str(product.get('dominant_color', '')),
                        'similarity_score': float(similarity_scores[idx])
                    })
            
            print(f"Generated {len(recommendations)} recommendations for {category}")
            return recommendations
            
        except Exception as e:
            print(f"Error in _get_category_recommendations for {category}: {e}")
            traceback.print_exc()
            return []

    def _get_trending_items(self, n_recommendations):
        """Get trending items from current or recent categories"""
        try:
            trending_items = []
            
            # If we have recent categories, prioritize them
            categories_to_use = self.current_categories or set(self.product_data.keys())
            
            if not categories_to_use:
                print("No categories available")
                return []
            
            items_per_category = max(2, n_recommendations // len(categories_to_use))
            
            for category in categories_to_use:
                df = self.product_data.get(category)
                if df is not None and len(df) > 0:
                    # Get top rated or recent items from this category
                    try:
                        # Sort by rating or date if available
                        if 'rating' in df.columns:
                            sorted_df = df.sort_values('rating', ascending=False)
                        elif 'date_added' in df.columns:
                            sorted_df = df.sort_values('date_added', ascending=False)
                        else:
                            sorted_df = df
                            
                        # Get top items
                        top_indices = sorted_df.index[:items_per_category]
                        
                        for idx in top_indices:
                            product = sorted_df.iloc[idx]
                            trending_items.append({
                                'product_id': str(product.get('product_id', '')),
                                'product_name': str(product.get('product_name', '')),
                                'brand': str(product.get('brand', 'Generic')),
                                'category_name': category,
                                'mrp': float(product.get('mrp', 0.0)),
                                'feature_image': str(product.get('feature_image', '')),
                                'description': str(product.get('product_details', '')),
                                'material_composition': str(product.get('material_composition', '')),
                                'style_attributes': str(product.get('style_attributes', '')),
                                'dominant_color': str(product.get('dominant_color', '')),
                                'similarity_score': 0.5  # Default score for trending items
                            })
                            
                    except Exception as e:
                        print(f"Error processing trending items for {category}: {e}")
                        continue
            
            print(f"Generated {len(trending_items)} trending items")
            return trending_items[:n_recommendations]
            
        except Exception as e:
            print(f"Error in _get_trending_items: {e}")
            traceback.print_exc()
            return []

    def _prepare_features(self, category):
        """Prepare TF-IDF features for a category"""
        try:
            if category not in self.tfidf_matrices:
                df = self.product_data.get(category)
                if df is not None:
                    # Combine relevant features
                    df['features'] = df.apply(
                        lambda x: ' '.join(filter(None, [
                            str(x.get('brand', '')),
                            str(x.get('style_attributes', '')),
                            str(x.get('material_composition', '')),
                            str(x.get('dominant_color', '')),
                            str(x.get('pattern_type', '')),
                            str(x.get('occasion', ''))
                        ])), axis=1
                    )
                    
                    # Create TF-IDF matrix
                    self.vectorizers[category] = TfidfVectorizer(stop_words='english')
                    self.tfidf_matrices[category] = self.vectorizers[category].fit_transform(df['features'])
                    print(f"Prepared features for {category}")
                    
        except Exception as e:
            print(f"Error preparing features for {category}: {e}")
            traceback.print_exc() 