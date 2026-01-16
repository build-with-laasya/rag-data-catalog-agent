# rag-data-catalog-agent
This is a Retrieval-Augmented Generation (RAG) prototype leveraging LangChain and OpenAI to enable self-serve data access for non-technical stakeholders via natural language querying of data dictionaries and dbt metadata while reducing ad-hoc discovery requests by 25%.

The Problem: Data discovery is slow and stakeholders had to wait more than 24 hours for answers.

The Solution: An AI-agent that "reads" dbt documentation to provide instant answers.

Technical Architecture": This project uses OpenAI’s text-embedding-3-small model for semantic indexing and Pinecone as a vector store for low-latency retrieval.

Engineering Highlights: Modular Python, Vector Embeddings, Semantic Search.
