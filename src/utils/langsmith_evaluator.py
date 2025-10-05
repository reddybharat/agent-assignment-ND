from langsmith import Client
from langsmith.evaluation import evaluate
from src.utils.retriever import Retriever
from langchain_google_genai import GoogleGenerativeAI

from dotenv import load_dotenv
load_dotenv()


class LangSmithEvaluator:
    def __init__(self, dataset_name: str = "RAG Dataset"):
        self.client = Client()
        self.dataset_name = dataset_name


    def create_custom_dataset(self):
        inputs = [
            "What are RAG?", 
            "What are different chunking strategies?",
            "Who is Elon Musk?"
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

            "The provided text does not contain information about Elon Musk. Therefore, I cannot answer the question."
        ]

        custom_dataset = self.client.create_dataset(
            dataset_name=self.dataset_name,
            description="A custom dataset for evaluation",
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
    

    def run_retrieval_evaluator(self):
        results = evaluate(
            Retriever().generate_response,
            data=self.dataset_name,
            evaluators=[self.similarity_evaluator],
            experiment_prefix="gemini_similarity_eval"
        )
        
        print("\nEvaluation completed!")
        print("Check LangSmith dashboard for detailed results.")
        
        return results


if __name__ == "__main__":
    evaluator = LangSmithEvaluator()
    # evaluator.create_custom_dataset()
    # print("Custom dataset created")
    evaluator.run_retrieval_evaluator()
