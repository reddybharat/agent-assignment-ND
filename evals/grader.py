from typing import Annotated
from pydantic import BaseModel, Field

class CorrectnessGrade(BaseModel):
    # Note that the order in the fields are defined is the order in which the model will generate them.
    # It is useful to put explanations before responses because it forces the model to think through
    # its final response before generating it:
    explanation: Annotated[str, Field(description="Explain your reasoning for the score")]
    correct: Annotated[bool, Field(description="True if the answer is correct, False otherwise.")]


class RelevanceGrade(BaseModel):
    explanation: Annotated[str, Field(description="Explain your reasoning for the relevance score")]
    relevant: Annotated[bool, Field(description="True if the answer is relevant to the question, False otherwise.")]


class GroundedGrade(BaseModel):
    explanation: Annotated[str, Field(description="Explain your reasoning for the groundedness score")]
    grounded: Annotated[bool, Field(description="True if the answer is grounded in the provided facts, False otherwise.")]


class RetrievalRelevanceGrade(BaseModel):
    explanation: Annotated[str, Field(description="Explain your reasoning for the retrieval relevance score")]
    relevant: Annotated[bool, Field(description="True if the retrieved documents are relevant to the question, False otherwise.")]
    