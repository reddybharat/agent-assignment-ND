import os
import re
from typing import List
from datetime import datetime
from pypdf import PdfReader
import qdrant_client
from qdrant_client import models
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv
load_dotenv()

class IngestPDF:
    def __init__(self, collection_name: str = "uploaded-pdfs"):
        # No API key needed for Hugging Face embeddings (completely free)
        
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if not qdrant_api_key:
            raise ValueError("QDRANT_API_KEY environment variable is not set. Please set it with your QDrant Cloud API key.")
        
        # Get QDrant Cloud configuration from environment
        qdrant_url = os.getenv("QDRANT_CLOUD_URL")
        
        self.client = qdrant_client.QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        self.collection_name = collection_name
        
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=50)
        self.embeddings = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Ingestion system initialized successfully.")

    def load_pdf(self, file_path):
        """
        Reads the text content from a PDF file and returns it as a single string.

        Parameters:
        - file_path (str): The file path to the PDF file.

        Returns:
        - str: The concatenated text content of all pages in the PDF.
        """
        try:
            # Logic to read pdf
            reader = PdfReader(file_path)

            # Loop over each page and store it in a variable
            text = ""
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text += page_text

            return text
        except Exception as e:
            raise

    def split_text(self, text: str):
        """
        Splits a text string into chunks using RecursiveCharacterTextSplitter.
        
        Parameters:
        - text (str): The input text to be split.

        Returns:
        - List[str]: A list containing text chunks.
        """
        # Use RecursiveCharacterTextSplitter to split the text
        chunks = self.splitter.split_text(text)
        return chunks

    def create_qdrant_db(self, documents: List, name: str = "uploaded-pdfs"):
        """
        Creates a Qdrant database using the provided documents and collection name.

        Parameters:
        - documents: An iterable of documents to be added to the Qdrant database.
        - name (str): The name of the collection within the Qdrant database.

        Returns:
        - Tuple[qdrant_client.Collection, str]: A tuple containing the created Qdrant Collection and its name.
        """
        try:
            # Remove the collection if it already exists
            try:
                self.client.delete_collection(name)
            except Exception as e:
                pass  # Ignore if it doesn't exist
            
            # Create new collection
            self.client.create_collection(
                name,
                vectors_config={
                    "text_embedding": models.VectorParams(
                        size=384,  # Dimension of all-MiniLM-L6-v2
                        distance=models.Distance.COSINE,
                    ),
                },
            )

            # Generate embeddings and create points
            texts = [doc['text'] for doc in documents]
            
            # Convert embeddings to list to avoid numpy array issues
            embeddings = self.embeddings.encode(texts, normalize_embeddings=True)
            embeddings = embeddings.tolist()  # Convert to list to avoid numpy array boolean issues
            
            points = []
            for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
                # Add metadata for better organization
                metadata = {
                    'filename': doc['filename'],
                    'source': doc['filename'],  # Add source for compatibility
                    'page': 1,  # Default page number
                    "chunk_index": idx,
                    "chunk_size": len(doc['text']),
                    "timestamp": str(datetime.now())
                }
                
                # Ensure embedding is a list of floats
                if not isinstance(embedding, list):
                    embedding = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
                
                # Ensure all values are floats
                embedding = [float(x) for x in embedding]
                
                point = {
                    "id": idx,
                    "vector": {"text_embedding": embedding},
                    "payload": {
                        "page_content": doc['text'],
                        "metadata": metadata
                    }
                }
                points.append(point)

            # Insert points into collection
            try:
                self.client.upsert(
                    collection_name=name,
                    points=points
                )
            except Exception as e:
                # Try inserting in smaller batches
                batch_size = 10
                for i in range(0, len(points), batch_size):
                    batch = points[i:i + batch_size]
                    try:
                        self.client.upsert(
                            collection_name=name,
                            points=batch
                        )
                    except Exception as batch_error:
                        continue

            try:
                collection = self.client.get_collection(name)
                return collection, name
            except Exception as e:
                return None, name
            
        except Exception as e:
            raise

    def run_ingestion_pipeline(self, file_paths: List[str]):
        """
        Runs the data ingestion pipeline

        Args:
            file_paths (List[str]): A list of file paths to the PDF files.
        """
        any_content = False
        total_chunks = 0
        all_chunks = []
        
        for i, file in enumerate(file_paths):
            try:
                # Load PDF content
                text = self.load_pdf(file)
                
                if not text or not text.strip():
                    continue
                
                # Split text into chunks
                chunked_text = self.split_text(text)
                
                if not chunked_text:
                    continue
                
                # Add file information to chunks
                file_name = os.path.basename(file)
                for j, chunk in enumerate(chunked_text):
                    all_chunks.append({
                        'text': chunk,
                        'filename': file_name,
                        'file_index': i,
                        'chunk_index': j
                    })
                
                total_chunks += len(chunked_text)
                any_content = True
                
            except Exception as e:
                continue
        
        if not any_content:
            error_msg = "No valid content found in any of the provided files"
            raise ValueError(error_msg)
        
        # Create QdrantDB collection with all chunks
        if all_chunks:
            try:
                collection, name = self.create_qdrant_db(all_chunks, "uploaded-pdfs")
                
                # Create and return vector store for compatibility
                from langchain_qdrant import QdrantVectorStore
                
                # Create a custom embedding function for LangChain compatibility
                from langchain.embeddings.base import Embeddings
                
                class SentenceTransformerEmbeddings(Embeddings):
                    def __init__(self, model):
                        self.model = model
                    
                    def embed_documents(self, texts):
                        embeddings = self.model.encode(texts, normalize_embeddings=True)
                        return embeddings.tolist()
                    
                    def embed_query(self, text):
                        embedding = self.model.encode([text], normalize_embeddings=True)
                        return embedding[0].tolist()
                
                embedding_function = SentenceTransformerEmbeddings(self.embeddings)
                
                vector_store = QdrantVectorStore(
                    client=self.client,
                    collection_name=name,
                    embedding=embedding_function,
                    vector_name="text_embedding"
                )
                
                print("PDF ingestion process completed successfully.")
                return vector_store
                
            except Exception as e:
                # Return a basic success indicator even if vector store creation fails
                return None
        
        return None

    def load_qdrant_collection(self, name: str = "uploaded-pdfs"):
        """
        Loads an existing Qdrant collection.

        Parameters:
        - name (str): The name of the collection to load.

        Returns:
        - qdrant_client.Collection: The loaded Qdrant collection.
        """
        try:
            collection = self.client.get_collection(name)
            return collection
        except Exception as e:
            raise