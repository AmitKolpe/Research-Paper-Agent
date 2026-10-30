"""Shared Hugging Face Inference API LLM factory (replaces OpenAI ChatOpenAI)."""

import os
from typing import TypeVar

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from pydantic import BaseModel

load_dotenv()

T = TypeVar("T", bound=BaseModel)


def get_llm(temperature: float = 0.2, max_new_tokens: int = 1024) -> ChatHuggingFace:
    """Return a chat model backed by the HF Inference API using HF_TOKEN / HF_MODEL."""
    endpoint = HuggingFaceEndpoint(
        repo_id=os.environ["HF_MODEL"],
        huggingfacehub_api_token=os.environ["HF_TOKEN"],
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        return_full_text=False,
    )
    return ChatHuggingFace(llm=endpoint)


def _append_json_instruction(inputs, instruction: str):
    """Append the JSON-format instruction to any supported chain input shape."""
    sys_msg = SystemMessage(content=instruction)
    if isinstance(inputs, ChatPromptValue):
        return ChatPromptValue(messages=[*inputs.messages, sys_msg])
    if isinstance(inputs, str):
        return [sys_msg, HumanMessage(content=inputs)]
    return [*inputs, sys_msg]


def with_pydantic_output(llm: ChatHuggingFace, schema: type[T]):
    """HF-compatible structured output.

    The HF Inference router rejects forced ``tool_choice``, so instead of
    OpenAI-style function calling we instruct the model to emit JSON matching
    the schema and parse it back into the Pydantic model.
    """
    parser = JsonOutputParser(pydantic_object=schema)
    instruction = (
        "CRITICAL OUTPUT FORMAT: Respond with ONLY a single valid JSON object "
        "that conforms exactly to this schema (no markdown, no code fences, "
        f"no explanation):\n{parser.get_format_instructions()}"
    )
    return (
        RunnableLambda(lambda inputs: _append_json_instruction(inputs, instruction))
        | llm
        | parser
        | RunnableLambda(lambda data: schema(**data))
    )
