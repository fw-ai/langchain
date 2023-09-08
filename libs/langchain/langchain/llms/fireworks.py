import os
from typing import Any, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import openai


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
