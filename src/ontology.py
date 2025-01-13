from neo4j import GraphDatabase
import pandas as pd
from pathlib import Path
import re
import time

class FashionOntologyBuilder:
    def __init__(self):
        # Update with your Neo4j Aura credentials
        URI = "neo4j+s://010bb698.databases.neo4j.io"
        AUTH = ("neo4j", "neo4j password")  # Your actual password
        
        try:
            self.driver = GraphDatabase.driver(URI, auth=AUTH)
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                result.single()
                print("Connected to Neo4j successfully!")
        except Exception as e:
            print(f"Connection error: {str(e)}")
            raise

    def check_existing_data(self, session):
        result = session.run("""
            MATCH (p:Product)
            RETURN count(p) as product_count
        """)
        return result.single()["product_count"]

    def get_processed_products(self, session):
        result = session.run("""
            MATCH (p:Product)
            RETURN p.product_id as product_id
        """)
        return {record["product_id"] for record in result}

    def load_and_build(self):
        with self.driver.session() as session:
            try:
                # Check existing data
                existing_count = self.check_existing_data(session)
                processed_products = self.get_processed_products(session)
                
                print(f"Found {existing_count} existing products")
                
                # Create constraint if not exists
                session.run("""
                    CREATE CONSTRAINT IF NOT EXISTS FOR (p:Product)
                    REQUIRE p.product_id IS UNIQUE
                """)
                
                files = {
                    'shirts': 'shirts_data_dump.csv',
                    'dresses': 'Dresses Data Dump.csv',
                    'jeans': 'Jeans Data Dump.csv',
                    'saree': 'Saree Data Dump.csv',
                    'sneakers': 'Sneakers Data Dump.csv',
                    'tshirts': 'tshirts_data_dump.csv',
                    'earrings': 'Earrings Data Dump.csv',
                    'kurtis': 'Data Dump Kurtis.csv'
                }
                
                base_path = Path(__file__).parent.parent.parent / 'frontend' / 'public' / 'Data'
                
                # Ensure categories exist
                for category in files.keys():
                    session.run("""
                        MERGE (c:Category {name: $category})
                    """, category=category)
                
                new_products = 0
                
                for category, filename in files.items():
                    file_path = base_path / filename
                    if not file_path.exists():
                        print(f"File not found: {filename}")
                        continue
                    
                    print(f"\nProcessing {filename}...")
                    df = pd.read_csv(file_path)
                    
                    # Filter out already processed products
                    df['product_id'] = df['product_id'].astype(str)
                    new_df = df[~df['product_id'].isin(processed_products)]
                    
                    if len(new_df) == 0:
                        print(f"All products from {filename} already processed")
                        continue
                    
                    print(f"Found {len(new_df)} new products in {category}")
                    
                    # Process in batches
                    batch_size = 50
                    for i in range(0, len(new_df), batch_size):
                        batch = new_df.iloc[i:i+batch_size]
                        records = []
                        
                        for _, row in batch.iterrows():
                            record = {
                                'product_id': str(row['product_id']),
                                'name': str(row.get('product_name', '')),
                                'price': float(row.get('mrp', 0)) if pd.notna(row.get('mrp')) else 0.0,
                                'brand': str(row.get('brand')) if pd.notna(row.get('brand')) else None,
                                'category': category
                            }
                            records.append(record)
                        
                        # Create new products and relationships
                        session.run("""
                            UNWIND $records as record
                            MERGE (p:Product {product_id: record.product_id})
                            SET p.name = record.name,
                                p.price = record.price
                            WITH p, record
                            MATCH (c:Category {name: record.category})
                            MERGE (p)-[:BELONGS_TO]->(c)
                            WITH p, record
                            FOREACH (brandName IN CASE WHEN record.brand IS NOT NULL 
                                THEN [record.brand] ELSE [] END |
                                MERGE (b:Brand {name: brandName})
                                MERGE (p)-[:MADE_BY]->(b)
                            )
                        """, {"records": records})
                        
                        new_products += len(records)
                        print(f"Processed {min(i + batch_size, len(new_df))} / {len(new_df)} new products")
                
                final_count = self.check_existing_data(session)
                print(f"\nSummary:")
                print(f"Previously existing products: {existing_count}")
                print(f"Newly added products: {new_products}")
                print(f"Total products now: {final_count}")
                
            except Exception as e:
                print(f"Error during data load: {str(e)}")
                raise

def create_ontology():
    try:
        builder = FashionOntologyBuilder()
        builder.load_and_build()
        print("Ontology update completed!")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    create_ontology()
