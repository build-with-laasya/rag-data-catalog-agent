import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from pinecone import Pinecone

load_dotenv()

# 1. Initialize Pinecone & OpenAI
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "data-catalog-embeddings"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 2. Load the JSON Data
with open('data/catalog.json', 'r') as f:
    raw_data = json.load(f)

# 3. Create "Document" objects
# We put the searchable text in 'page_content' and extra details in 'metadata'
docs = []
for item in raw_data:
    doc = Document(
        page_content=item["content"],
        metadata=item["metadata"]
    )
    docs.append(doc)

# 4. Perform the Ingestion (The "Upsert")
print(f"Starting ingestion of {len(docs)} tables into Pinecone...")
vectorstore = PineconeVectorStore.from_documents(
    docs, 
    embeddings, 
    index_name=index_name
)
print("Ingestion complete!")

