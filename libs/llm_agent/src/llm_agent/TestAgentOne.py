from typing import Optional

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from .BaseLLMAgent import BaseLLMAgent as LLMEnabledAgent


class TestAgentOne(LLMEnabledAgent):

    def __init__(self, prompt: str, llm: Optional, name: str = "TestAgentOne"):
        _prompt = PromptTemplate(
            template=(prompt),
            input_variables=["transcript_text", "diarized_transcript"],
        )
        super().__init__(name=name, prompt_template=_prompt, output_parser=None, llm=llm)

    def handle(self, transcript_text: str = "", diarized_transcript: str = ""):
        print(f"[{self.name}] working ...")
        # print(self.log_promt({"transcript_text": transcript_text, "diarized_transcript": diarized_transcript}))
        return self.generate_responseV3({
            "transcript_text": transcript_text,
            "diarized_transcript": diarized_transcript
        })
