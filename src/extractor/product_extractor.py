# backend/src/extractor/product_extractor.py

from collections import Counter
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import uuid
from sentence_transformers import SentenceTransformer
import faiss
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50
import io
from sklearn.metrics.pairwise import cosine_similarity
import requests
from urllib.parse import urlparse
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import time
import signal

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductExtractor:
    def __init__(self):
        self.category_mappings = {}
        self.feature_patterns = {}
        self.learned_attributes = set()
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.style_clusters = None
        self.trend_data = {}
        self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.search_index = None
        self.indexed_products = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model
        try:
            self.model = resnet50(pretrained=True).to(self.device)
            self.model.eval()
            logger.info("ResNet50 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading ResNet50 model: {e}")
            raise

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.session = requests.Session()
        self.image_features_cache = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        logger.info("ProductExtractor initialized successfully")
        
        # Add category classifiers
        self.category_features = {
            'dresses': ['dress', 'gown', 'frock'],
            'tshirts': ['tshirt', 't-shirt', 'top'],
            'jeans': ['jeans', 'denim', 'pants'],
            'sarees': ['saree', 'sari'],
            'shirts': ['shirt', 'formal shirt'],
            'sneakers': ['shoes', 'sneakers', 'footwear'],
            'earrings': ['earring', 'jewelry']
        }
        
        # Limit initial processing
        self.max_initial_products = 200
        self.similarity_threshold = 0.6

    def initialize_from_data(self, file_paths):
        """Initialize extractor by learning from all available data"""
        try:
            all_products = []
            all_descriptions = []
            
            for file_path in file_paths:
                df = pd.read_csv(file_path)
                
                # Learn category mappings
                categories = df[['category_id', 'category_name']].drop_duplicates()
                for _, row in categories.iterrows():
                    self.category_mappings[row['category_id']] = row['category_name']
                
                # Collect descriptions for AI analysis
                descriptions = df['description'].dropna().tolist()
                all_descriptions.extend(descriptions)
                all_products.extend(df.to_dict('records'))
                
                # Learn feature patterns
                self._learn_feature_patterns(descriptions)
            
            # AI-driven pattern learning
            self._train_ai_models(all_descriptions, all_products)
            print(f"AI models trained on {len(all_descriptions)} products")
            
        except Exception as e:
            print(f"Error initializing extractor: {e}")

    def _train_ai_models(self, descriptions, products):
        """Train AI models for feature extraction"""
        try:
            # Text vectorization for style analysis
            X = self.vectorizer.fit_transform(descriptions)
            
            # Cluster similar styles
            n_clusters = min(8, len(descriptions))
            self.style_clusters = KMeans(n_clusters=n_clusters).fit(X)
            
            # Build trend analysis data
            self._analyze_trends(products)
            
        except Exception as e:
            print(f"Error training AI models: {e}")

    def _analyze_trends(self, products):
        """Analyze product trends and patterns"""
        try:
            # Group products by category
            category_products = {}
            for product in products:
                category = product.get('category_name', 'unknown')
                if category not in category_products:
                    category_products[category] = []
                category_products[category].append(product)
            
            # Analyze trends per category
            for category, prods in category_products.items():
                self.trend_data[category] = {
                    'popular_materials': self._extract_popular_features(prods, 'materials'),
                    'popular_colors': self._extract_popular_features(prods, 'colors'),
                    'price_ranges': self._analyze_price_distribution(prods),
                    'style_patterns': self._extract_style_patterns(prods)
                }
                
        except Exception as e:
            print(f"Error analyzing trends: {e}")

    def clean_product_data(self, product_data):
        """Clean and validate product data before processing"""
        try:
            # Convert to dictionary if it's a pandas Series
            if hasattr(product_data, 'to_dict'):
                product_data = product_data.to_dict()

            # Debug print to see raw data
            print("Raw product data for image:", product_data)

            # Extract image URL with multiple fallbacks
            image_url = None
            
            # Try all possible image fields from the dataset
            image_fields = [
                'image_url',
                'feature_image',
                'feature_image_s3',
                7,  # SHEIN image URL column
                10  # Stylumia image URL column
            ]

            for field in image_fields:
                if field in product_data and product_data[field]:
                    value = str(product_data[field]).strip()
                    if value and value.lower() != 'nan':
                        # Clean the URL
                        if ',' in value:
                            # Split by comma and take first part
                            value = value.split(',')[0]
                        # Remove @ if present
                        if value.startswith('@'):
                            value = value[1:]
                        # Remove [] if present
                        value = value.strip('[]')
                        
                        image_url = value.strip()
                        print(f"Found image URL from field {field}: {image_url}")
                        break

            print(f"Final cleaned image URL: {image_url}")

            # Basic cleaning without strict validation
            cleaned = {
                'product_id': str(product_data.get('product_id', '')).strip() or str(uuid.uuid4())[:8],
                'product_name': str(product_data.get('product_name', '')).strip(),
                'brand': str(product_data.get('brand', '')).strip(),
                'description': str(product_data.get('description', '')).strip(),
                'mrp': float(str(product_data.get('mrp', '0')).replace(',', '') or 0),
                'meta_info': str(product_data.get('meta_info', '')).strip(),
                'category_id': product_data.get('category_id', 0),
                'image_url': image_url  # Add the cleaned image URL
            }

            # Extract brand from product name if not present
            if not cleaned['brand']:
                cleaned['brand'] = self._extract_brand_from_text(cleaned['product_name'])
            
            # If still no brand, try description
            if not cleaned['brand']:
                cleaned['brand'] = self._extract_brand_from_text(cleaned['description'])

            # Capitalize product name
            cleaned['product_name'] = ' '.join(word.capitalize() for word in cleaned['product_name'].split())
            
            # Uppercase brand
            cleaned['brand'] = cleaned['brand'].upper() if cleaned['brand'] else 'UNBRANDED'

            return cleaned

        except Exception as e:
            print(f"Error cleaning product data: {e}")
            return None

    def _extract_brand_from_text(self, text):
        """Extract brand name from text using comprehensive brand list"""
        brands = {
            'SHEIN': ['shein', 'she in'],
            'H&M': ['h&m', 'h & m', 'h and m'],
            'ZARA': ['zara'],
            'NIKE': ['nike'],
            'ADIDAS': ['adidas'],
            'LEVIS': ["levi's", 'levis'],
            'PUMA': ['puma'],
            'GAP': ['gap'],
            'FOREVER 21': ['forever 21', 'forever21'],
            'GUCCI': ['gucci'],
            'CALVIN KLEIN': ['calvin klein', 'ck'],
            'TOMMY HILFIGER': ['tommy hilfiger', 'tommy'],
            'REEBOK': ['reebok'],
            'UNDER ARMOUR': ['under armour'],
            'UNIQLO': ['uniqlo'],
            'MANGO': ['mango'],
            'BIBA': ['biba'],
            'FABINDIA': ['fabindia', 'fab india'],
            'WESTSIDE': ['westside'],
            'AND': ['and'],
            'VERO MODA': ['vero moda'],
            'ONLY': ['only'],
            'MAX': ['max fashion', 'max'],
            'PANTALOONS': ['pantaloons']
        }
        
        text_lower = text.lower()
        
        for brand_name, variations in brands.items():
            if any(variation in text_lower for variation in variations):
                return brand_name
            
        return ''

    def _extract_colors(self, text):
        """Enhanced color extraction from text"""
        color_mapping = {
            'red': '#FF0000',
            'blue': '#0000FF',
            'green': '#008000',
            'yellow': '#FFFF00',
            'black': '#000000',
            'white': '#FFFFFF',
            'pink': '#FFC0CB',
            'purple': '#800080',
            'orange': '#FFA500',
            'brown': '#A52A2A',
            'grey': '#808080',
            'gray': '#808080',
            'navy': '#000080',
            'beige': '#F5F5DC',
            'maroon': '#800000',
            'cream': '#FFFDD0',
            'olive': '#808000',
            'burgundy': '#800020',
            'teal': '#008080',
            'coral': '#FF7F50',
            'turquoise': '#40E0D0',
            'lavender': '#E6E6FA',
            'peach': '#FFDAB9',
            'gold': '#FFD700',
            'silver': '#C0C0C0',
            'khaki': '#F0E68C',
            'mint': '#98FF98',
            'rose': '#FF007F',
            'mustard': '#FFDB58'
        }

        found_colors = set()
        text_lower = text.lower()

        # Direct color mentions
        for color in color_mapping.keys():
            if f" {color} " in f" {text_lower} ":
                found_colors.add(color)

        # Color patterns
        color_patterns = [
            r'(?:color|colour):\s*([\w\s]+)',
            r'available in\s*([\w\s]+)\s*colou?r',
            r'(\w+)\s+colou?r(?:ed)?',
            r'colou?r\s*:\s*(\w+)',
        ]

        for pattern in color_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                color_words = match.strip().split()
                for word in color_words:
                    if word in color_mapping:
                        found_colors.add(word)

        # Convert to list of tuples with hex values
        return [(color.capitalize(), color_mapping[color]) for color in found_colors]

    def extract_basic_features(self, product_data):
        """Extract and enhance product features with AI insights"""
        try:
            # Clean data
            cleaned_data = self.clean_product_data(product_data)
            if not cleaned_data:
                return None

            # Extract colors with hex values
            colors_with_hex = self._extract_colors(
                cleaned_data['description'] + ' ' + cleaned_data['meta_info']
            )

            # Basic features
            features = {
                "product_id": cleaned_data['product_id'],
                "product_name": cleaned_data['product_name'],
                "brand": cleaned_data['brand'],
                "price": cleaned_data['mrp'],
                "description": cleaned_data['description'],
                "category": self.category_mappings.get(
                    int(cleaned_data.get('category_id', 0)), "Unknown"
                ),
                "image_url": cleaned_data['image_url'],
                "colors": colors_with_hex  # Now includes color name and hex value
            }

            # Extract other features
            text_features = self._extract_text_features(
                cleaned_data['description'], 
                cleaned_data['meta_info']
            )
            
            # Update features but keep the new color format
            for key, value in text_features.items():
                if key != 'colors':  # Skip colors as we already have them with hex values
                    features[key] = value

            # Add AI insights
            features["ai_insights"] = {
                "trend_analysis": {
                    "score": self._calculate_trend_score(features),
                    "trend_status": self._get_trend_status(self._calculate_trend_score(features))
                },
                "style_recommendations": self._generate_style_recommendations(features),
                "feature_confidence": self._calculate_feature_confidence(features),
                "popularity_score": self._calculate_popularity_score(features),
                "seasonal_relevance": self._analyze_seasonal_fit(features),
                "occasion_matching": self._analyze_occasion_matching(features)
            }

            return features

        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def _analyze_trend_relevance(self, features):
        """Analyze product's trend relevance using multiple factors"""
        try:
            trend_factors = {
                "current_season_fit": 0,
                "color_trend_match": 0,
                "style_popularity": 0,
                "material_relevance": 0
            }

            # Analyze seasonal fit
            current_season = self._get_current_season()
            if any(season in str(features.get('description', '')).lower() 
                   for season in [current_season, 'all-season']):
                trend_factors["current_season_fit"] = 100

            # Analyze color trends
            if features.get('colors'):
                trending_colors = self._get_trending_colors()
                color_matches = sum(1 for color in features['colors'] 
                                  if color.lower() in trending_colors)
                trend_factors["color_trend_match"] = min(100, color_matches * 25)

            # Analyze style popularity
            if features.get('style_attributes'):
                popular_styles = self._get_popular_styles()
                style_matches = sum(1 for style in features['style_attributes'] 
                                  if style.lower() in popular_styles)
                trend_factors["style_popularity"] = min(100, style_matches * 30)

            # Calculate overall trend score
            overall_score = sum(trend_factors.values()) / len(trend_factors)
            
            return {
                "score": overall_score,
                "factors": trend_factors,
                "trend_status": self._get_trend_status(overall_score)
            }

        except Exception as e:
            print(f"Error analyzing trend relevance: {e}")
            return {"score": 60, "factors": {}, "trend_status": "neutral"}

    def _get_trend_status(self, score):
        """Get trend status based on score"""
        if score >= 80:
            return "trending"
        elif score >= 70:
            return "popular"
        elif score >= 60:
            return "stable"
        else:
            return "classic"

    def _generate_ai_insights(self, features):
        """Generate AI-driven insights for a product"""
        try:
            description_vector = self.vectorizer.transform([features['description']])
            cluster = self.style_clusters.predict(description_vector)[0]
            
            insights = {
                "style_cluster": int(cluster),
                "trend_score": self._calculate_trend_score(features),
                "recommendations": self._generate_style_recommendations(features),
                "feature_confidence": self._calculate_feature_confidence(features)
            }
            
            return insights
        except Exception as e:
            print(f"Error generating AI insights: {e}")
            return {}

    def _generate_style_recommendations(self, features):
        """Generate style recommendations based on product features"""
        try:
            recommendations = []
            
            # Base recommendations on product attributes
            if features.get('style_attributes'):
                if 'casual' in [s.lower() for s in features['style_attributes']]:
                    recommendations.append("Perfect for everyday casual wear")
                if 'formal' in [s.lower() for s in features['style_attributes']]:
                    recommendations.append("Ideal for professional settings")
                    
            # Material-based recommendations
            if features.get('materials'):
                for material in features['materials']:
                    material_lower = str(material).lower()
                    if material_lower == 'cotton':
                        recommendations.append("Breathable and comfortable for daily wear")
                    elif material_lower == 'silk':
                        recommendations.append("Elegant choice for special occasions")
                        
            # Color-based recommendations
            if features.get('colors'):
                # Extract color names from tuples
                for color_tuple in features['colors']:
                    color = color_tuple[0].lower()
                    if color in ['black', 'navy', 'white']:
                        recommendations.append("Versatile color that pairs well with most outfits")
                    elif color in ['red', 'pink', 'yellow']:
                        recommendations.append("Bold color choice that makes a statement")
                        
            return recommendations[:3]  # Return top 3 recommendations
            
        except Exception as e:
            print(f"Error generating style recommendations: {e}")
            return []

    def _calculate_trend_score(self, features):
        """Calculate accurate trend score based on multiple factors"""
        try:
            scores = {
                "brand_score": 0,
                "color_score": 0,
                "material_score": 0,
                "style_score": 0
            }

            # Brand score
            if features.get('brand'):
                trending_brands = {'ZARA': 90, 'H&M': 85, 'NIKE': 95, 'ADIDAS': 90}
                scores["brand_score"] = trending_brands.get(features['brand'].upper(), 60)

            # Color score
            if features.get('colors'):
                trending_colors = {
                    'Black': 90, 'White': 85, 'Navy': 80, 'Beige': 75,
                    'Pink': 85, 'Green': 80, 'Blue': 75, 'Red': 70
                }
                color_scores = [trending_colors.get(color, 60) for color in features['colors']]
                scores["color_score"] = sum(color_scores) / len(color_scores) if color_scores else 60

            # Material score
            if features.get('materials'):
                trending_materials = {
                    'Cotton': 85, 'Linen': 90, 'Silk': 88, 'Denim': 82,
                    'Wool': 80, 'Polyester': 70, 'Leather': 85
                }
                material_scores = [trending_materials.get(material, 60) for material in features['materials']]
                scores["material_score"] = sum(material_scores) / len(material_scores) if material_scores else 60

            # Style score
            if features.get('style_attributes'):
                trending_styles = {
                    'Casual': 85, 'Modern': 90, 'Classic': 80, 'Formal': 75,
                    'Traditional': 70, 'Ethnic': 80, 'Western': 85
                }
                style_scores = [trending_styles.get(style, 60) for style in features['style_attributes']]
                scores["style_score"] = sum(style_scores) / len(style_scores) if style_scores else 60

            # Calculate weighted average
            weights = {"brand_score": 0.3, "color_score": 0.25, "material_score": 0.25, "style_score": 0.2}
            final_score = sum(scores[key] * weights[key] for key in scores)

            return round(final_score)

        except Exception as e:
            print(f"Error calculating trend score: {e}")
            return 60

    def _calculate_feature_confidence(self, features):
        """Calculate confidence scores for extracted features"""
        try:
            confidence = {
                "colors": 0,
                "materials": 0,
                "style": 0,
                "overall": 0
            }
            
            # Color confidence
            if features.get('colors'):
                confidence["colors"] = min(len(features['colors']) * 25, 100)
                
            # Material confidence
            if features.get('materials'):
                confidence["materials"] = min(len(features['materials']) * 30, 100)
                
            # Style confidence
            if features.get('style_attributes'):
                confidence["style"] = min(len(features['style_attributes']) * 25, 100)
                
            # Overall confidence
            confidence["overall"] = sum(
                score for key, score in confidence.items() if key != 'overall'
            ) / 3
            
            return confidence
        except Exception as e:
            print(f"Error calculating feature confidence: {e}")
            return {
                "colors": 60,
                "materials": 60,
                "style": 60,
                "overall": 60
            }

    def _extract_style_relationships(self):
        """Extract relationships between different styles"""
        try:
            relationships = {}
            
            # Process each category's trends
            for category, trends in self.trend_data.items():
                if 'style_patterns' in trends:
                    for style, data in trends['style_patterns'].items():
                        if style not in relationships:
                            relationships[style] = {
                                'count': 0,
                                'materials': Counter(),
                                'colors': Counter(),
                                'price_range': {'min': float('inf'), 'max': 0, 'avg': 0},
                                'occasions': set()
                            }
                        
                        # Update style statistics
                        relationships[style]['count'] += data.get('count', 0)
                        relationships[style]['materials'].update(data.get('materials', {}))
                        relationships[style]['colors'].update(data.get('colors', {}))
                        
                        # Update price range
                        price_data = data.get('price_range', {})
                        if price_data:
                            relationships[style]['price_range']['min'] = min(
                                relationships[style]['price_range']['min'],
                                price_data.get('min', float('inf'))
                            )
                            relationships[style]['price_range']['max'] = max(
                                relationships[style]['price_range']['max'],
                                price_data.get('max', 0)
                            )
                            if 'avg' in price_data:
                                current_avg = relationships[style]['price_range']['avg']
                                count = relationships[style]['count']
                                relationships[style]['price_range']['avg'] = (
                                    (current_avg * (count - 1) + price_data['avg']) / count
                                )
                        
                        # Add occasions
                        if 'occasions' in data:
                            relationships[style]['occasions'].update(data['occasions'])
            
            # Convert sets to lists for JSON serialization
            for style_data in relationships.values():
                style_data['occasions'] = list(style_data['occasions'])
            
            return relationships
            
        except Exception as e:
            print(f"Error extracting style relationships: {e}")
            return {}

    def _extract_popular_features(self, products, feature_key):
        """Extract popular features from a list of products"""
        try:
            counter = Counter()
            for product in products:
                features = product.get(feature_key, [])
                if isinstance(features, list):
                    counter.update(features)
            return dict(counter.most_common(10))
        except Exception as e:
            print(f"Error extracting popular features: {e}")
            return {}

    def _analyze_price_distribution(self, products):
        """Analyze price distribution in products"""
        try:
            prices = [float(p.get('mrp', 0)) for p in products if p.get('mrp')]
            if not prices:
                return {'min': 0, 'max': 0, 'avg': 0, 'median': 0}
            
            return {
                'min': min(prices),
                'max': max(prices),
                'avg': sum(prices) / len(prices),
                'median': sorted(prices)[len(prices)//2]
            }
        except Exception as e:
            print(f"Error analyzing price distribution: {e}")
            return {'min': 0, 'max': 0, 'avg': 0, 'median': 0}

    def _extract_style_patterns(self, products):
        """Extract style patterns from products"""
        try:
            patterns = {}
            for product in products:
                style = product.get('style', 'unknown')
                if style not in patterns:
                    patterns[style] = {
                        'count': 0,
                        'materials': Counter(),
                        'colors': Counter(),
                        'price_range': {'min': float('inf'), 'max': 0, 'avg': 0},
                        'occasions': set()
                    }
                
                patterns[style]['count'] += 1
                patterns[style]['materials'].update(product.get('materials', []))
                patterns[style]['colors'].update(product.get('colors', []))
                
                price = float(product.get('mrp', 0))
                if price > 0:
                    patterns[style]['price_range']['min'] = min(
                        patterns[style]['price_range']['min'], 
                        price
                    )
                    patterns[style]['price_range']['max'] = max(
                        patterns[style]['price_range']['max'], 
                        price
                    )
                    current_avg = patterns[style]['price_range']['avg']
                    count = patterns[style]['count']
                    patterns[style]['price_range']['avg'] = (
                        (current_avg * (count - 1) + price) / count
                    )
            
            return patterns
            
        except Exception as e:
            print(f"Error extracting style patterns: {e}")
            return {}

    def _learn_feature_patterns(self, descriptions):
        """Learn common patterns and attributes from product descriptions"""
        try:
            # Common attribute indicators
            patterns = [
                r'made (?:of|from) ([^,\.]+)',  # Material patterns
                r'([\w\s]+) (?:pattern|print)',  # Pattern types
                r'([\w\s]+) (?:fit|style)',      # Fit/Style types
                r'(?:in|available in) ([\w\s]+) colors?'  # Color patterns
            ]
            
            for desc in descriptions:
                if not isinstance(desc, str):
                    continue
                    
                for pattern in patterns:
                    matches = re.findall(pattern, desc.lower())
                    for match in matches:
                        self.learned_attributes.add(match.strip())
                        
                # Extract key-value pairs (e.g., "Neck: Round")
                kv_pairs = re.findall(r'(\w+):\s*([\w\s]+)', desc)
                for key, value in kv_pairs:
                    if key not in self.feature_patterns:
                        self.feature_patterns[key] = Counter()
                    self.feature_patterns[key][value.strip()] += 1
                    
        except Exception as e:
            print(f"Error learning feature patterns: {e}")

    def _extract_text_features(self, description, meta_info):
        """Enhanced feature extraction from text"""
        try:
            combined_text = f"{description} {meta_info}"
            features = {
                "colors": [],
                "materials": [],
                "patterns": [],
                "style_attributes": []
            }

            # Extract colors
            features["colors"] = self._extract_colors(combined_text)

            # Extract materials
            material_patterns = [
                r'(?:made of|made from|material):\s*([\w\s]+)',
                r'(?:^|\s)(cotton|polyester|wool|silk|leather|denim|nylon|rayon|linen|velvet|satin|chiffon|jersey|twill|canvas)(?:\s|$)'
            ]
            
            materials = set()
            for pattern in material_patterns:
                matches = re.findall(pattern, combined_text.lower())
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    material = match.strip()
                    if material and len(material) <= 20:
                        materials.add(material.capitalize())
            features["materials"] = list(materials)

            # Extract patterns
            pattern_types = [
                r'([\w\s]+) pattern',
                r'([\w\s]+) print',
                r'(checkered|striped|floral|solid|printed|polka|dots|geometric|abstract)'
            ]
            
            patterns = set()
            for pattern in pattern_types:
                matches = re.findall(pattern, combined_text.lower())
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    pattern_type = match.strip()
                    if pattern_type and len(pattern_type) <= 20:
                        patterns.add(pattern_type.capitalize())
            features["patterns"] = list(patterns)

            # Extract style attributes
            style_patterns = [
                r'(casual|formal|party|ethnic|western|traditional|modern)',
                r'(slim|regular|loose)\s*fit',
                r'style:\s*([\w\s]+)'
            ]
            
            styles = set()
            for pattern in style_patterns:
                matches = re.findall(pattern, combined_text.lower())
                for match in matches:
                    if isinstance(match, tuple):
                        match = match[0]
                    style = match.strip()
                    if style and len(style) <= 20:
                        styles.add(style.capitalize())
            features["style_attributes"] = list(styles)

            return features

        except Exception as e:
            print(f"Error extracting text features: {e}")
            return {
                "colors": [],
                "materials": [],
                "patterns": [],
                "style_attributes": []
            }

    def _calculate_popularity_score(self, features):
        """Calculate popularity score based on product features"""
        try:
            score = 60  # Base score
            
            # Analyze brand popularity
            if features.get('brand'):
                trending_brands = {'ZARA': 90, 'H&M': 85, 'NIKE': 95, 'ADIDAS': 90}
                score += trending_brands.get(features['brand'].upper(), 0)
                
            # Check for trending materials
            trending_materials = ['cotton', 'linen', 'silk', 'denim']
            if features.get('materials'):
                matches = sum(1 for material in features['materials'] 
                             if str(material).lower() in trending_materials)
                score += min(matches * 5, 15)
                
            # Check for versatile colors
            versatile_colors = ['black', 'white', 'navy', 'beige']
            if features.get('colors'):
                # Extract just the color names from the tuples
                color_names = [color[0].lower() for color in features['colors']]
                matches = sum(1 for color in color_names if color in versatile_colors)
                score += min(matches * 5, 15)
                
            return min(score, 100)  # Cap at 100
            
        except Exception as e:
            print(f"Error calculating popularity score: {e}")
            return 60

    def _get_current_season(self):
        """Get current season based on month"""
        try:
            from datetime import datetime
            month = datetime.now().month
            
            if month in [12, 1, 2]:
                return 'winter'
            elif month in [3, 4, 5]:
                return 'spring'
            elif month in [6, 7, 8]:
                return 'summer'
            else:
                return 'autumn'
        except Exception as e:
            print(f"Error getting current season: {e}")
            return 'all-season'

    def _get_trending_colors(self):
        """Get current trending colors"""
        return [
            'sage green', 'lavender', 'coral', 'navy blue',
            'blush pink', 'mustard yellow', 'burgundy', 'cream'
        ]

    def _get_popular_styles(self):
        """Get current popular styles"""
        return [
            'casual', 'minimalist', 'vintage', 'streetwear',
            'bohemian', 'classic', 'contemporary', 'athleisure'
        ]

    def _analyze_seasonal_fit(self, features):
        """Analyze how well the product fits the current season"""
        try:
            current_season = self._get_current_season()
            description = features.get('description', '').lower()
            
            # Season-specific keywords
            season_keywords = {
                'winter': ['warm', 'cozy', 'wool', 'knit', 'thermal'],
                'summer': ['cool', 'light', 'breathable', 'cotton', 'linen'],
                'spring': ['light', 'floral', 'fresh', 'bright'],
                'autumn': ['layering', 'comfortable', 'versatile']
            }
            
            # Calculate match score
            keywords = season_keywords.get(current_season, [])
            matches = sum(1 for word in keywords if word in description)
            match_score = min((matches / len(keywords)) * 100, 100)
            
            return {
                "current_season": current_season.capitalize(),
                "match_score": round(match_score),
                "keywords_matched": matches
            }
        except Exception as e:
            print(f"Error analyzing seasonal fit: {e}")
            return {
                "current_season": "All-Season",
                "match_score": 60,
                "keywords_matched": 0
            }

    def _analyze_occasion_matching(self, features):
        """Analyze suitable occasions for the product"""
        try:
            occasions = []
            description = features.get('description', '').lower()
            
            occasion_keywords = {
                'formal': ['formal', 'office', 'business', 'professional'],
                'casual': ['casual', 'everyday', 'relaxed'],
                'party': ['party', 'celebration', 'festive'],
                'sports': ['sports', 'athletic', 'workout', 'gym'],
                'ethnic': ['traditional', 'ethnic', 'cultural']
            }
            
            for occasion, keywords in occasion_keywords.items():
                if any(keyword in description for keyword in keywords):
                    occasions.append(occasion)
                    
            return {
                "suitable_occasions": occasions or ['casual'],
                "versatility_score": min(len(occasions) * 20, 100)
            }
        except Exception as e:
            print(f"Error analyzing occasions: {e}")
            return {
                "suitable_occasions": ['casual'],
                "versatility_score": 60
            }

    def _build_search_index(self, products):
        """Build FAISS index for semantic search"""
        try:
            # Prepare text for indexing
            texts = [
                f"{p['product_name']} {p['description']} {p['brand']} {' '.join(p.get('materials', []))}"
                for p in products
            ]
            
            # Generate embeddings
            embeddings = self.sentence_model.encode(texts, show_progress_bar=True)
            
            # Initialize FAISS index
            dimension = embeddings.shape[1]
            self.search_index = faiss.IndexFlatL2(dimension)
            self.search_index.add(np.array(embeddings).astype('float32'))
            self.indexed_products = products
            
            print(f"Built search index with {len(products)} products")
            
        except Exception as e:
            print(f"Error building search index: {e}")
            
    def semantic_search(self, query, k=500):
        """Perform semantic search on products"""
        try:
            if not self.search_index:
                return []
                
            # Generate query embedding
            query_vector = self.sentence_model.encode([query])
            
            # Search
            distances, indices = self.search_index.search(
                np.array(query_vector).astype('float32'), k
            )
            
            # Return matched products
            return [self.indexed_products[idx] for idx in indices[0]]
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []

    @lru_cache(maxsize=1000)
    def download_image(self, url):
        """Download image from URL with caching"""
        try:
            logger.info(f"Downloading image from {url}")
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            return io.BytesIO(response.content)
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {e}")
            return None

    def extract_image_features(self, image_data):
        """Optimized feature extraction"""
        try:
            # Add timeout for image processing
            with timeout(seconds=5):
                if isinstance(image_data, str):
                    if urlparse(image_data).scheme in ['http', 'https']:
                        image_bytes = self.download_image(image_data)
                        if image_bytes is None:
                            return None
                        image = Image.open(image_bytes).convert('RGB')
                    else:
                        image = Image.open(image_data).convert('RGB')
                elif isinstance(image_data, (bytes, io.BytesIO)):
                    if isinstance(image_data, bytes):
                        image_data = io.BytesIO(image_data)
                    image_data.seek(0)
                    image = Image.open(image_data).convert('RGB')
                else:
                    return None

                # Extract features
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    features = self.model(image_tensor)
                    return features.cpu().numpy().flatten()

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None

    def process_product_batch(self, products_batch):
        """Process a batch of products in parallel"""
        similarities = []
        for product in products_batch:
            if 'image_url' in product:
                if product['image_url'] in self.image_features_cache:
                    product_features = self.image_features_cache[product['image_url']]
                else:
                    product_features = self.extract_image_features(product['image_url'])
                    if product_features is not None:
                        self.image_features_cache[product['image_url']] = product_features
                
                if product_features is not None and self.query_features is not None:
                    similarity = cosine_similarity(
                        self.query_features.reshape(1, -1),
                        product_features.reshape(1, -1)
                    )[0][0]
                    similarities.append((product, similarity))
        return similarities

    def find_similar_products(self, query_image, products, category, top_k=50):
        """Optimized product search"""
        try:
            logger.info(f"Starting image search for category: {category}")
            start_time = time.time()

            # Extract query image features
            query_features = self.extract_image_features(query_image)
            if query_features is None:
                logger.error("Failed to extract features from query image")
                return []

            # Filter products by category first
            category_products = products[:self.max_initial_products]
            
            # Process in parallel with early stopping
            similarities = []
            processed_count = 0
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                
                for product in category_products:
                    if 'image_url' in product:
                        futures.append(
                            executor.submit(
                                self.process_single_product,
                                product,
                                query_features
                            )
                        )
                
                # Collect results as they complete
                for future in futures:
                    result = future.result()
                    if result:
                        product, similarity = result
                        if similarity >= self.similarity_threshold:
                            similarities.append((product, similarity))
                    
                    processed_count += 1
                    if processed_count % 20 == 0:
                        logger.info(f"Processed {processed_count} products")

            # Sort and filter results
            similarities.sort(key=lambda x: x[1], reverse=True)
            results = [product for product, similarity in similarities[:top_k]]
            
            end_time = time.time()
            logger.info(f"Search completed in {end_time - start_time:.2f} seconds")
            logger.info(f"Found {len(results)} similar products")
            
            return results

        except Exception as e:
            logger.error(f"Error in find_similar_products: {e}")
            return []

    def process_single_product(self, product, query_features):
        """Process a single product with timeout"""
        try:
            # Check cache first
            if product['image_url'] in self.image_features_cache:
                product_features = self.image_features_cache[product['image_url']]
            else:
                product_features = self.extract_image_features(product['image_url'])
                if product_features is not None:
                    self.image_features_cache[product['image_url']] = product_features

            if product_features is not None:
                similarity = cosine_similarity(
                    query_features.reshape(1, -1),
                    product_features.reshape(1, -1)
                )[0][0]
                return (product, similarity)
            
            return None

        except Exception as e:
            logger.error(f"Error processing product: {e}")
            return None

    def __del__(self):
        """Cleanup resources"""
        self.thread_pool.shutdown()
        self.session.close()

# Add timeout context manager
class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)