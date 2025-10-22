from langsmith import Client
from langsmith.evaluation import aevaluate
from langchain_google_genai import GoogleGenerativeAI

from src.utils.retriever import Retriever
from src.graphs.builder import build_graph
from src.graphs.type import RAGAgentState

import asyncio
from dotenv import load_dotenv
load_dotenv()


class LangSmithEvaluator:
    def __init__(self, dataset_name: str = "assignment-langgraph-dataset"):
        self.client = Client()
        self.dataset_name = dataset_name

    def get_custom_dataset(self):
        inputs = [
            "What are RAG?", 
            "What are different chunking strategies?",
            "Who is Elon Musk?",
            "What is the weather in London?"
        ]

        outputs = [
            "RAG stands for Retrieval-Augmented Generation, a framework that combines information retrieval systems with generative language models to produce more accurate and contextually relevant responses. The core of RAG architecture comprises two main components working in synergy: the retriever and generation components. The retriever component fetches relevant information from a predefined knowledge base, ensuring the AI system has access to up-to-date and accurate information, while the generation component uses the retrieved information to produce coherent and contextually relevant responses.",
            """Chunking strategies involve dividing large documents into smaller segments (chunks). The strategies include:
            *   **Fixed-size chunking**: This defines a specific number of tokens per chunk and includes overlap between chunks to minimize semantic context loss. It is computationally cheap and simple to implement.
                *   **Example:** Langchain CharacterTextSplitter
            *   **Context-aware chunking**: This leverages the intrinsic structure of the text for more meaningful and contextually relevant chunks.
                *   **Sentence Splitting**: Aligns with models optimized for embedding sentence-level content.
                    *   **Naive Splitting**: Basic method using periods and newlines.
                    *   **NLTK (Natural Language Toolkit)**: A comprehensive Python library for language processing with a sentence tokenizer.""",

            "The provided text does not contain information about Elon Musk. Therefore, I cannot answer the question.",
            "The weather in London is few clouds with a temperature of 18.5°C."
        ]

        return inputs, outputs

    def create_custom_dataset(self):

        inputs, outputs = self.get_custom_dataset()

        custom_dataset = self.client.create_dataset(
            dataset_name=self.dataset_name,
            description="custom dataset for evaluation",
        )

        self.client.create_examples(
            inputs=[{"question": q} for q in inputs],
            outputs=[{"answer": a} for a in outputs],
            dataset_id=custom_dataset.id,
        )

    # Custom evaluators
    async def similarity_evaluator(self, outputs: dict, reference_outputs: dict) -> bool:
        """Similarity evaluator - returns True only if outputs are identical"""
        judge_llm = GoogleGenerativeAI(model="gemini-2.0-flash")

        # Combine instructions and user message into a single prompt
        prompt = f"""Given an actual answer and an expected answer, determine whether the actual answer contains all of the information in the expected answer. 
        Respond with 'CORRECT' if the actual answer does contain all of the expected information and 'INCORRECT' otherwise. Do not include anything else in your response.

        ACTUAL ANSWER: {outputs}

        EXPECTED ANSWER: {reference_outputs}

        Response:"""
        
        response = await judge_llm.ainvoke(prompt)

        return response.strip().upper() == "CORRECT"
    
    async def run_retrieval_evaluator(self):
        results = aevaluate(
            Retriever().generate_response,
            data=self.dataset_name,
            evaluators=[self.similarity_evaluator],
            max_concurrency=3,
            experiment_prefix="gemini_similarity_eval"
        )
        
        print("\nEvaluation completed!")
        print("Check LangSmith dashboard for detailed results.")
        
        return results

    async def graph_similarity_evaluator(self, outputs: dict, reference_outputs: dict) -> bool:
        """Similarity evaluator - returns True only if outputs are identical"""

        judge_llm = GoogleGenerativeAI(model="gemini-2.0-flash")

        # Combine instructions and user message into a single prompt
        prompt = f"""Given an actual answer and an expected answer, determine whether the actual answer contains all of the information in the expected answer.
        If its a weather query, you just need to check the format of the answer. It should be like this "The weather in London is few clouds with a temperature of 18.64°C."
        Respond with 'CORRECT' if the actual answer does contain all of the expected information and 'INCORRECT' otherwise. Do not include anything else in your response.

        ACTUAL ANSWER: {outputs.get('answer')}

        EXPECTED ANSWER: {reference_outputs}

        Response:"""
        
        response = await judge_llm.ainvoke(prompt)

        return response.strip().upper() == "CORRECT"

    async def run_graph_evaluator(self):
        # Langgraph graphs are also langchain runnables.
        def example_to_state(inputs: dict) -> dict:
            return {
                "query": inputs['question'],
                "answer": "",
                "status": "Pending",
                "is_weather_query": False,
                "location": ""
            }

        app = build_graph(RAGAgentState)

        target = example_to_state | app
        experiment_results = await aevaluate(
                    target,
                    data=self.dataset_name,
                    evaluators=[self.graph_similarity_evaluator],
                    max_concurrency=4,  # optional
                    experiment_prefix="gemini_graph_similarity_eval",  # optional
                )


if __name__ == "__main__":
    evaluator = LangSmithEvaluator()

    # evaluator.create_custom_dataset()
    # print("Custom dataset created")

    # print("Running retrieval evaluator")
    # asyncio.run(evaluator.run_retrieval_evaluator())

    print("Running graph evaluator")
    asyncio.run(evaluator.run_graph_evaluator())