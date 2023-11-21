import os
import chromadb

# We are handling some chromadb settings that will help us create the embeddings and store them in a parquet format through duckdb
client = chromadb.PersistentClient(path="db")
