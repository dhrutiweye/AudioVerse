"""BaseAgent class for creating agent implementations."""
from typing import Any, Dict, Optional, Union, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


class BaseLLMAgent:
    def __init__(
            self,
            name: str,
            llm: Optional[object] = None,
            prompt_template: Optional[PromptTemplate] = None,
            output_parser: Optional[object] = None,
    ):
        self.name = name
        self.prompt_template = prompt_template
        self.output_parser = output_parser
        if not self.output_parser:
            self.output_parser = StrOutputParser()  # Fallback for plain text

        self.llm = llm or ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5,
                                                 google_api_key='AIzaSyB6r1cXq_c6w99AVRzahIaoSL3OZjD8hrg')

    def generate_response(self, prompt: str):
        chain = self.llm | self.output_parser
        return chain.invoke(prompt)

    def log_promt(self, prompt: Union[str, Dict[str, str]]):
        if self.prompt_template and isinstance(prompt, dict):
            formatted_prompt = self.prompt_template.format(**prompt)
            return formatted_prompt
        else:
            return prompt

    def generate_responseV2(self, prompt: Union[str, Dict[str, str]]):
        chain = self.llm | self.output_parser

        if self.prompt_template and isinstance(prompt, dict):
            formatted_prompt = self.prompt_template.format(**prompt)
            # print(formatted_prompt)
            return chain.invoke(formatted_prompt)

        # Fallback for string-based prompt
        if isinstance(prompt, str):
            return chain.invoke(prompt)

        raise ValueError("Invalid prompt type. Expected str or dict based on usage.")

    def generate_response_with_history(self, chat_history: List[Dict], user_input: str) -> str:
        """
        Generate a contextual response using chat history + user input.
        """
        full_prompt = ""

        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                full_prompt += f"[System]: {content}\n"
            elif role == "user":
                full_prompt += f"[User]: {content}\n"
            elif role == "assistant":
                full_prompt += f"[Assistant]: {content}\n"

        # full_prompt += f"[User]: {user_input}\n[Assistant]:"
        # New JSON-output mode
        if self.prompt_template:
            prompt = self.prompt_template.format(
                history=full_prompt,
                user_input=user_input
            )
        else:
            # Fallback to string prompt
            prompt = f"{full_prompt}[User]: {user_input}\n[Assistant]:"

        return self.generate_response(full_prompt)