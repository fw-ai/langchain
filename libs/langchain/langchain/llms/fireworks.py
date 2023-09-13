import os
from typing import Any, Dict, Iterator, List, Optional

import backoff

from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
)
from langchain.chat_models.openai import _create_retry_decorator
from langchain.llms.base import LLM
from langchain.schema.language_model import LanguageModelInput
from langchain.schema.output import GenerationChunk
from langchain.schema.runnable.config import RunnableConfig
import openai


def _stream_response_to_generation_chunk(
    stream_response: Dict[str, Any],
) -> GenerationChunk:
    """Convert a stream response to a generation chunk."""
    return GenerationChunk(
        text=stream_response["choices"][0]["text"],
        generation_info=dict(
            finish_reason=stream_response["choices"][0].get("finish_reason", None),
            logprobs=stream_response["choices"][0].get("logprobs", None),
        ),
    )


class Fireworks(LLM):
    """Fireworks models."""

    model = "accounts/fireworks/models/llama-v2-7b-chat"
    model_kwargs: Optional[dict] = {"temperature": 0.7, "max_tokens": 512, "top_p": 1}
    fireworks_api_base: Optional[str] = "https://api.fireworks.ai/inference/v1"
    fireworks_api_key: Optional[str] = None
    max_retries: int = 20

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fireworks"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        params = {
            "model": self.model,
            "prompt": prompt,
            **self.model_kwargs,
        }
        response = self.completion_with_retry(**params)

        return response["choices"][0]["text"]

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        params = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            **self.model_kwargs,
        }
        for stream_resp in self.completion_with_retry(**params):
            chunk = _stream_response_to_generation_chunk(stream_resp)
            yield chunk

    def stream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        prompt = self._convert_input(input).to_string()
        generation: Optional[GenerationChunk] = None
        for chunk in self._stream(prompt):
            yield chunk.text
            if generation is None:
                generation = chunk
            else:
                generation += chunk
        assert generation is not None

    def completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            return openai.Completion.create(
                api_base="https://api.fireworks.ai/inference/v1",
                api_key=os.environ.get("FIREWORKS_API_KEY"),
                **kwargs,
            )

        return _completion_with_retry(**kwargs)
