import qdrant_client
from qdrant_client.models import Distance, PointStruct, VectorParams


from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

import google.generativeai as gemini_client

import os
from typing import List
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

class IngestPDF:
    def __init__(self, collection_name: str = "uploaded-pdfs"):
        
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if not qdrant_api_key:
            raise ValueError("QDRANT_API_KEY environment variable is not set. Please set it with your QDrant Cloud API key.")
        
        qdrant_url = os.getenv("QDRANT_CLOUD_URL")
        
        self.client = qdrant_client.QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        self.collection_name = collection_name
        
        #gemini_client utilized for embeddings
        gemini_client.configure(api_key=os.getenv("GEMINI_API_KEY"))

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
        

    def create_qdrant_db(self, documents: List):
        """
        Creates a Qdrant database using the provided documents and collection name.

        Parameters:
        - documents: An iterable of document chunks to be added to the Qdrant database.
        - name (str): The name of the collection within the Qdrant database.

        Returns:
        - Tuple[qdrant_client.Collection, str]: A tuple containing the created Qdrant Collection and its name.
        """
        try:
            # Create collection if it doesn't exist
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    self.collection_name,
                    vectors_config=VectorParams(
                        size=768, # Dimension of gemini-embedding-001
                        distance=Distance.COSINE,
                    ),
                )

            # Get collection details to check for points id, so that upsert doesnt update existing points
            collection_details = self.client.get_collection(self.collection_name)
            if collection_details:
                points_count = collection_details.points_count
            else:
                points_count = 0

            # Generate embeddings for each document
            embeddings  = [
                        gemini_client.embed_content(
                            model="models/embedding-001",
                            content=doc['text'],
                            task_type="retrieval_document",
                            title="Qdrant x Gemini",
                        )
                        for doc in documents
            ]
            
            # Create list of points
            points = []
            for idx, (response, doc) in enumerate(zip(embeddings, documents)):
                points_count += 1
                metadata = {
                    'source': doc['filename'],
                    "chunk_size": len(doc['text']),
                    "timestamp": str(datetime.now())
                }
                for key, value in doc['metadata'].items():
                    metadata[key] = value

                point = PointStruct(
                    id=points_count,
                    vector=response['embedding'],
                    payload={"page_content": doc['text'], "metadata": metadata},
                )
                points.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
        except Exception as e:
            raise

    def run_ingestion_pipeline(self, file_paths: List[str]):
        """
        Runs the data ingestion pipeline

        Args:
            file_paths (List[str]): A list of file paths to the PDF files.
        """
        try:
            any_content = False
            total_chunks = 0
            all_chunks = []
            
            for i, file in enumerate(file_paths):
                try:

                    chunked_text = self.docling_load_and_split(file)
                    
                    if not chunked_text:
                        continue
                    
                    file_name = os.path.basename(file)
                    for j, chunk in enumerate(chunked_text):
                        if chunk.page_content and chunk.page_content.strip():
                            chunk_data_dict = {
                                'text': chunk.page_content,
                                'filename': file_name,
                                'file_index': i,
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
            self.create_qdrant_db(all_chunks)
            
        except Exception as e:
            raise