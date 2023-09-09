import os
from typing import Any, Dict, List, Mapping, Optional, Tuple
from langchain.adapters.openai import convert_dict_to_message, convert_message_to_dict

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import BaseChatModel, SimpleChatModel
from langchain.llms.base import LLM
from langchain.schema.messages import BaseMessage
from langchain.schema.output import ChatGeneration, ChatResult
import openai


class ChatFireworks(BaseChatModel):
    """Fireworks Chat models."""

    model = "accounts/fireworks/models/llama-v2-13b-chat"
    model_kwargs: Optional[dict] = {"temperature": 0.7, "max_tokens": 512, "top_p": 1}
    fireworks_api_url: Optional[str] = "https://api.fireworks.ai/inference/v1"
    fireworks_api_key: Optional[str] = os.environ.get("FIREWORKS_API_KEY")

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fireworks-chat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts = self._create_message_dicts(messages, stop)
        response = openai.ChatCompletion.create(
            api_base=self.fireworks_api_url,
            api_key=self.fireworks_api_key,
            model=self.model,
            messages=message_dicts,
            **self.model_kwargs,
        )
        return self._create_chat_result(response)

    def _combine_llm_outputs(self, llm_outputs: List[Optional[dict]]) -> dict:
        return llm_outputs[0]

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for res in response["choices"]:
            message = convert_dict_to_message(res["message"])
            gen = ChatGeneration(
                message=message,
                generation_info=dict(finish_reason=res.get("finish_reason")),
            )
            generations.append(gen)
        llm_output = {"model": self.model}
        return ChatResult(generations=generations, llm_output=llm_output)

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]]]:
        message_dicts = [convert_message_to_dict(m) for m in messages]
        return message_dicts
