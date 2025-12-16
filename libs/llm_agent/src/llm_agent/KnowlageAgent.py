import json
from typing import Optional, Dict, List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from .BaseLLMAgent import BaseLLMAgent as LLMEnabledAgent

class Chunk(BaseModel):
    id: int = Field(description="Unique numeric id for the chunk")
    category: str = Field(description="Category name this fact belongs to")
    text: str = Field(description="Atomic natural-language fact extracted from the JSON input")


class KnowlageAgent(LLMEnabledAgent):
    class ChunkList(BaseModel):
        chunks: List[Chunk] = Field(description="List of chunk objects extracted from the JSON")

    ChunkList.model_rebuild()
    parser = JsonOutputParser(pydantic_object=ChunkList)
    prompt = PromptTemplate(
        template=("""
                you are an expert knowledge-base architect.
I will give you a JSON object containing company knowledge, product information, sales processes, and operational details.

Your task is to convert the JSON into a list of memory chunks, each containing only one atomic fact, written in natural language, and optimized for vector retrieval and Mem0-style memory storage.

Rules for Chunk Creation

Each chunk must contain exactly one fact.
Write chunks in clear natural language, not JSON.
Each chunk must be standalone, with enough context to understand it without the original JSON.
Each chunk must include a category tag such as:
    "company_profile"
    "operational_info"
    "communication_standards"
    "Product_catalog"
    "sales_service_strategy"

Output MUST be valid JSON:
    {format_instructions}
Do NOT summarize.
Do NOT combine multiple facts.
Include every detail from the input JSON.
Ensure the JSON is valid and parsable.

Break everything into atomic, Mem0-compatible memory sentences.
NO explanation, NO extra text — only the JSON.
Now I will give you my JSON knowledge base. Convert it into chunks exactly as instructe


Now I will give you the JSON. Convert it into chunks exactly as instructed. {company_data}
                """),
        input_variables=["company_data"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    def __init__(self, llm: Optional, name: str = "KnowlageAgent"):
        super().__init__(name=name, prompt_template=self.prompt, output_parser=self.parser, llm=llm)

    def handle(self, company_data: Dict = "") -> (str, ChunkList):
        print(f"[{self.name}] working ...")
        return self.generate_responseV3({
            "company_data": json.dumps(company_data),
        })
