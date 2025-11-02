"""AgentModel package for agent base classes."""
from .SignleQueriAgent import SignleQueriAgent
from langchain_openai import ChatOpenAI

__all__ = [
    "SignleQueriAgent",
    "ChatOpenAI"
]

