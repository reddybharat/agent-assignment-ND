from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import Client
from evals import prompts, grader
from src.utils.retriever import Retriever
from langsmith.evaluation import aevaluate, evaluate
from src.graphs.builder import build_graph
from src.graphs.type import RAGAgentState
import asyncio

class Eval:
    def __init__(self, dataset_name: str = "assignment-langgraph-dataset"):
        self.dataset_name = dataset_name
        self.langsmith_client = Client()

    # Correctness: Response vs reference answer
    # how similar/correct is the RAG answer, relative to a ground-truth answer
    def correctness(self, inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
        """An evaluator for RAG answer accuracy"""
        correctness_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0).with_structured_output(grader.CorrectnessGrade)
        # with_structured_output() forces LangSmith to return response in specified format
        answers = f"""\
                    QUESTION: {inputs['question']}
                    GROUND TRUTH ANSWER: {reference_outputs['answer']}
                    STUDENT ANSWER: {outputs['answer']}"""
        grade = correctness_llm.invoke([
                {"role": "system", "content": prompts.correctness_instructions}, 
                {"role": "user", "content": answers}
        ])
        return grade.correct
    
    # Relevance: Response vs input
    # how well does the generated response address the initial user input
    def relevance(self, inputs: dict, outputs: dict) -> bool:
        """An evaluator for RAG answer relevance"""
        relevance_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0).with_structured_output(grader.RelevanceGrade)
        answer = f"QUESTION: {inputs['question']}\nSTUDENT ANSWER: {outputs['answer']}"
        grade = relevance_llm.invoke([
            {"role": "system", "content": prompts.relevance_instructions}, 
            {"role": "user", "content": answer}
        ])
        return grade.relevant

    # Groundedness: Response vs retrieved docs
    # to what extent does the generated response agree with the retrieved context
    def grounded(self, inputs: dict, outputs: dict) -> bool:
        """An evaluator for RAG answer groundedness"""
        grounded_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0).with_structured_output(grader.GroundedGrade)
        doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
        answer = f"FACTS: {doc_string}\nSTUDENT ANSWER: {outputs['answer']}"
        grade = grounded_llm.invoke([{"role": "system", "content": prompts.grounded_instructions}, {"role": "user", "content": answer}])
        return grade.grounded

    # Retrieval relevance: Retrieved docs vs input
    # how relevant are my retrieved results for this query
    def retrieval_relevance(self, inputs: dict, outputs: dict) -> bool:
        """An evaluator for RAG answer retrieval relevance"""
        retrieval_relevance_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0).with_structured_output(grader.RetrievalRelevanceGrade)
        doc_string = "\n\n".join(doc.page_content for doc in outputs["documents"])
        answer = f"FACTS: {doc_string}\nQUESTION: {inputs['question']}"
        # Run evaluator
        grade = retrieval_relevance_llm.invoke([
            {"role": "system", "content": prompts.retrieval_relevance_instructions}, 
            {"role": "user", "content": answer}
        ])
        return grade.relevant

    async def run_graph_evaluator(self):
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
            evaluators=[self.correctness, self.grounded, self.relevance, self.retrieval_relevance],
            experiment_prefix="rag-doc-relevance",
            max_concurrency=4,
        )

        print("\nRAG Graph Evaluation completed!")

        return experiment_results
    
    def retriever_wrapper(self, inputs: dict) -> dict:
        """Wrapper function to convert retriever output to expected format"""
        retriever = Retriever()
        
        # Get the response from retriever
        response = retriever.generate_response(inputs['question'])
        
        # Get the documents that were used for retrieval
        docs = retriever.retrieve(inputs['question'])
        
        # Convert docs to the format expected by evaluators
        formatted_docs = []
        for doc in docs:
            # Create a simple object with page_content attribute
            class Document:
                def __init__(self, page_content, metadata):
                    self.page_content = page_content
                    self.metadata = metadata
            
            formatted_docs.append(Document(doc['page_content'], doc['metadata']))
        
        return {
            "answer": response,
            "documents": formatted_docs
        }

    def run_retrieval_evaluator(self):
        experiment_results = evaluate(
            self.retriever_wrapper,
            data=self.dataset_name,
            evaluators=[self.correctness, self.grounded, self.relevance, self.retrieval_relevance],
            max_concurrency=3,
            experiment_prefix="retriever_evals"
        )

        print("\nRAG Retrieval Evaluation completed!")

        return experiment_results

if __name__ == "__main__":
    retrieval_eval_results = Eval().run_retrieval_evaluator()
    graph_eval_results = Eval().run_graph_evaluator()

    print("Check LangSmith dashboard for detailed results.")
