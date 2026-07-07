"""LLM backend factory.

Both dev (Ollama) and prod (vLLM) expose an OpenAI-compatible /v1 endpoint, so
a single OpenAILLMService pointed at LLM_BASE_URL covers both. Tokens stream
out of the box, which is what keeps TTS starting on the first sentence.
"""

from __future__ import annotations

from loguru import logger

from pipecat.services.openai.llm import OpenAILLMService

from realtime_ai.config import Settings


def create_llm(settings: Settings) -> OpenAILLMService:
    logger.info(
        f"LLM backend: {settings.resolved_llm_backend} "
        f"model={settings.llm_model} base_url={settings.llm_base_url}"
    )
    return OpenAILLMService(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        params=OpenAILLMService.InputParams(
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        ),
    )
