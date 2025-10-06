import qdrant_client
from qdrant_client import models
from langchain_qdrant import QdrantVectorStore

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

import os
import logging
from typing import List
from datetime import datetime
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
        
        self.embeddings = SentenceTransformer('all-MiniLM-L6-v2')

    def docling_load_and_split(self, file_path):
        try:
            loader = DoclingLoader(
                    file_path=file_path,
                    export_type=ExportType.MARKDOWN
            )
            docs = loader.load()

            if not docs:
                return []
            
            md_splitter = MarkdownHeaderTextSplitter(
                        headers_to_split_on=[
                            ("#", "Header_1"),
                            ("##", "Header_2"),
                            ],
                        )
            md_splits = [split for doc in docs for split in md_splitter.split_text(doc.page_content)]

            chunk_size = 2000
            chunk_overlap = 50
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            splits = text_splitter.split_documents(md_splits)

            return splits
            
        except Exception as e:
            return []
        

    def create_qdrant_db(self, documents: List, name: str = "uploaded-pdfs"):
        """
        Creates a Qdrant database using the provided documents and collection name.

        Parameters:
        - documents: An iterable of document chunks to be added to the Qdrant database.
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
                metadata = {
                    'source': doc['filename'],
                    "chunk_index": idx,
                    "chunk_size": len(doc['text']),
                    "timestamp": str(datetime.now())
                }

                for key, value in doc['metadata'].items():
                    metadata[key] = value
                
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

                chunked_text = self.docling_load_and_split(file)
                
                if not chunked_text:
                    continue
                
                # Add file information to chunks
                file_name = os.path.basename(file)
                for j, chunk in enumerate(chunked_text):
                    if chunk.page_content and chunk.page_content.strip():
                        chunk_data_dict = {
                            'text': chunk.page_content,
                            'filename': file_name,
                            'file_index': i,
                            'chunk_index': j,
                            'metadata': chunk.metadata
                        }
                        

                        all_chunks.append(chunk_data_dict)

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
                
                vector_store = QdrantVectorStore(
                    client=self.client,
                    collection_name=name,
                    embedding=self.embeddings,
                    vector_name="text_embedding"
                )             
                return vector_store
                
            except Exception as e:
                return None
        return None