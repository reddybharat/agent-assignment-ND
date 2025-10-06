from langchain_google_genai import GoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import SentenceTransformer
import qdrant_client
from src.utils.prompts import RETRIEVER_PROMPT
import os

from dotenv import load_dotenv
load_dotenv()

class Retriever:
    def __init__(self, collection_name: str = "uploaded-pdfs"):
        # Get QDrant Cloud configuration from environment
        qdrant_url = os.getenv("QDRANT_CLOUD_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not qdrant_api_key:
            raise ValueError("QDRANT_API_KEY environment variable is not set. Please set it with your QDrant Cloud API key.")
        
        self.client = qdrant_client.QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        self.collection_name = collection_name
        
        self.embeddings = SentenceTransformer('all-MiniLM-L6-v2')
        
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
        
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=embedding_function,
            vector_name="text_embedding"
        )
        
        self.llm = GoogleGenerativeAI(model="gemini-2.0-flash")

    def retrieve(self, query: str, k: int = 3):
        """Retrieve top-k similar chunks from Qdrant"""
        results = self.vector_store.similarity_search(query, k=k)
        
        # If page_content is empty, try to get content from metadata
        for result in results:
            if not result.page_content and result.metadata:
                # Check if content is stored in metadata
                if 'text' in result.metadata:
                    result.page_content = result.metadata['text']
                elif 'content' in result.metadata:
                    result.page_content = result.metadata['content']
        
        return results

    def generate_response(self, query: str, k: int = 3) -> str:
        """Retrieve context and generate a response with Gemini 2 Flash"""
        docs = self.retrieve(query, k=k)
        
        context = "\n\n".join(
            [f"[{d.metadata.get('source')} - page {d.metadata.get('page')}]\n{d.page_content}" for d in docs]
        )

        # Create prompt with context using the imported prompt template
        prompt = RETRIEVER_PROMPT.format(context=context, query=query)

        response = self.llm.invoke(prompt)
        return response