from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

import os
from constants import client

persist_directory = 'db'

def main():
    for root, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
                
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    
    # Creating text embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") #Huggingface api creates problems sometimes, so this is being used
    
    # Creating vector stores here
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory,client_settings=client)
    db.persist()
    db = None
    
    if __name__ == "__main__":
        main()
        
        
    # Import all the required libraries
    # Defining directory
    # Created a main function
    # Loading the document
    # Passing it to a Recursivetextsplitter
    # Creating the embeddings and storing it into the chromadb database