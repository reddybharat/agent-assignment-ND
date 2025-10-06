from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as gemini_client

import qdrant_client
from src.utils.prompts import RETRIEVER_PROMPT
import os

from dotenv import load_dotenv
load_dotenv()

class Retriever:
    def __init__(self, collection_name: str = "uploaded-pdfs"):

        self.qdrant_url = os.getenv("QDRANT_CLOUD_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not self.qdrant_api_key:
            raise ValueError("QDRANT_API_KEY environment variable is not set. Please set it with your QDrant Cloud API key.")
        
        self.client = qdrant_client.QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
        )
        self.collection_name = collection_name
        
        self.llm = GoogleGenerativeAI(model="gemini-2.0-flash")
        
        gemini_client.configure(api_key=os.getenv("GEMINI_API_KEY"))

    def retrieve(self, query: str, k: int = 7):
        """Retrieve top-k similar chunks from Qdrant"""
        # results = self.vector_store.similarity_search(query, k=k)
        results = self.client.search(
                collection_name=self.collection_name,
                query_vector=gemini_client.embed_content(
                    model="models/embedding-001",
                    content=query,
                    task_type="retrieval_query",
                )["embedding"],
                limit=k,
            )

        formatted_results = []
        for result in results:       
            formatted_results.append({
                'page_content': result.payload.get('page_content', ''),
                'metadata': result.payload.get('metadata', {})
            })
        
        return formatted_results

    def generate_response(self, query: str, k: int = 7) -> str:
        """Retrieve context and generate a response with Gemini 2 Flash"""
        docs = self.retrieve(query, k=k)
        
        context = "\n\n".join(
            [f"[Metadata - {d['metadata']}]\n{d['page_content']}" for d in docs]
        )

        # Create prompt with context using the imported prompt template
        prompt = RETRIEVER_PROMPT.format(context=context, query=query)

        response = self.llm.invoke(prompt)
        return response