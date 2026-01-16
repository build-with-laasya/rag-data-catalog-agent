import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA

load_dotenv()

# 1. Setup Data
with open('data/catalog.json') as f:
    data = json.load(f)

documents = [
    Document(page_content=f"Table: {item['table']}. Description: {item['description']}. Columns: {', '.join(item['columns'])}")
    for item in data
]

# 2. Initialize Vector DB
pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "data-catalog" # Ensure this index exists in your Pinecone console

# 3. Create Search Index
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone.from_documents(documents, embeddings, index_name=index_name)

# 4. RAG Chain
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vectorstore.as_retriever()
)

# 5. User Interaction
if __name__ == "__main__":
    query = "I need to find a table for patient identity verification. Which one should I use?"
    result = qa_chain.invoke(query)
    print(f"\nAI Response: {result['result']}")
