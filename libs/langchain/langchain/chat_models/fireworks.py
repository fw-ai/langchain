import os
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple
from langchain.adapters.openai import convert_dict_to_message, convert_message_to_dict

from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.schema.messages import (
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessageChunk,
    FunctionMessageChunk,
    HumanMessageChunk,
    SystemMessageChunk,
)
from langchain.schema.output import ChatGeneration, ChatGenerationChunk, ChatResult
import openai


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("role")
    content = _dict.get("content") or ""
    if _dict.get("function_call"):
        additional_kwargs = {"function_call": dict(_dict["function_call"])}
    else:
        additional_kwargs = {}

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"])
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(content=content)


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

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message_dicts = self._create_message_dicts(messages, stop)
        response = await openai.ChatCompletion.acreate(
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

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts = self._create_message_dicts(messages, stop)
        default_chunk_class = AIMessageChunk
        for chunk in openai.ChatCompletion.create(
            api_base=self.fireworks_api_url,
            api_key=self.fireworks_api_key,
            model=self.model,
            messages=message_dicts,
            stream=True,
            **self.model_kwargs,
        ):
            choice = chunk["choices"][0]
            chunk = _convert_delta_to_message_chunk(
                choice["delta"], default_chunk_class
            )
            finish_reason = choice.get("finish_reason")
            generation_info = (
                dict(finish_reason=finish_reason) if finish_reason is not None else None
            )
            default_chunk_class = chunk.__class__
            yield ChatGenerationChunk(message=chunk, generation_info=generation_info)
