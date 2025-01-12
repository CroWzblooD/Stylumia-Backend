from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from ontology import FashionOntologyBuilder

# Create a separate FastAPI app for ontology
ontology_app = FastAPI()

# Add CORS middleware
ontology_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ontology builder
ontology_builder = FashionOntologyBuilder()

@ontology_app.on_event("startup")
async def startup_event():
    # Initialize ontology using the new method name
    ontology_builder.load_and_build()
    print("Ontology loaded successfully!")

@ontology_app.get("/fashion-trends")
def get_fashion_trends():
    with ontology_builder.driver.session() as session:
        result = session.run("""
            MATCH (c:Category)<-[:BELONGS_TO]-(p:Product)
            WITH c.name as category, count(p) as products
            RETURN category, products
            ORDER BY products DESC
        """)
        return {"trends": [dict(record) for record in result]}

if __name__ == "__main__":
    uvicorn.run("ontology_server:ontology_app", host="0.0.0.0", port=8001, reload=True) 