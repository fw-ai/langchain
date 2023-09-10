import os
from typing import Any, Dict, Iterator, List, Optional

from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
)
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

    model = "accounts/fireworks/models/llama-v2-13b-chat"
    model_kwargs: Optional[dict] = {"temperature": 0.7, "max_tokens": 512, "top_p": 1}
    fireworks_api_url: Optional[str] = "https://api.fireworks.ai/inference/v1"
    fireworks_api_key: Optional[str] = os.environ.get("FIREWORKS_API_KEY")

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
        response = openai.Completion.create(
            api_base=self.fireworks_api_url,
            api_key=self.fireworks_api_key,
            model=self.model,
            prompt=prompt,
            **self.model_kwargs,
        )
        return response["choices"][0]["text"]

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        for stream_resp in openai.Completion.create(
            api_base=self.fireworks_api_url,
            api_key=self.fireworks_api_key,
            model=self.model,
            prompt=prompt,
            stream=True,
            **self.model_kwargs,
        ):
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
