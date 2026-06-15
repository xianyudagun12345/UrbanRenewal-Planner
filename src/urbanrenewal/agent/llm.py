"""Shared LLM factory for UrbanRenewal agents and tools."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.urbanrenewal.config import cfg


def build_llm(temperature: float = 0.3) -> ChatOpenAI:
    """Create the DashScope OpenAI-compatible chat model used by the app."""
    return ChatOpenAI(
        model=cfg.llm_model,
        api_key=cfg.dashscope_api_key,
        base_url=cfg.dashscope_base_url,
        temperature=temperature,
    )
