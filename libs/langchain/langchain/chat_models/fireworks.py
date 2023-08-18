"""Wrapper around Fireworks APIs."""
from __future__ import annotations

import requests
import json
import logging
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from pydantic import Field, root_validator
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    ChatGeneration,
    ChatResult,
)
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.utils import get_from_dict_or_env

if TYPE_CHECKING:
    import tiktoken

logger = logging.getLogger(__name__)


def update_token_usage(
    keys: Set[str], response: Dict[str, Any], token_usage: Dict[str, Any]
) -> None:
    """Update token usage."""
    _keys_to_use = keys.intersection(response["usage"])
    for _key in _keys_to_use:
        if _key not in token_usage:
            token_usage[_key] = response["usage"][_key]
        else:
            token_usage[_key] += response["usage"][_key]


def _update_response(response: Dict[str, Any], stream_response: Dict[str, Any]) -> None:
    """Update response from the stream response."""
    response["choices"][0]["text"] += stream_response["choices"][0]["text"]
    response["choices"][0]["finish_reason"] = stream_response["choices"][0].get(
        "finish_reason", None
    )
    response["choices"][0]["logprobs"] = stream_response["choices"][0]["logprobs"]


async def acompletion_with_retry(
    llm: Union[BaseFireworks, FireworksChat], **kwargs: Any
) -> Any:
    """Use tenacity to retry the async completion call."""
    answers = []
    result = execute(kwargs["messages"], kwargs["model"], llm.fireworks_api_key)
    curr_string = json.loads(result)["choices"][0]["message"]
    answers.append(curr_string)

    return answers


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        content = _dict["content"] or ""
        if _dict.get("function_call"):
            additional_kwargs = {"function_call": dict(_dict["function_call"])}
        else:
            additional_kwargs = {}
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=_dict["content"])
    elif role == "function":
        return FunctionMessage(content=_dict["content"], name=_dict["name"])
    else:
        return ChatMessage(content=_dict["content"], role=role)


def execute(messages, model: str, api_key: Optional[str]) -> Any:
    """Execute LLM query"""
    requestUrl = "https://api.fireworks.ai/inference/v1/chat/completions"
    requestBody = {
        "model": model,
        "messages": messages,
    }
    requestHeaders = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(requestUrl, headers=requestHeaders, json=requestBody)
    return response.text


def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    else:
        raise ValueError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


class ChatFireworks(BaseChatModel):
    """Wrapper around Fireworks Chat large language models.

    Get the environment variable ``FIREWORKS_API_KEY`` set with your API key.

    Example:
        .. code-block:: python

            from langchain.chat_models import ChatFireworks
            fireworks = ChatFireworks(model_name="accounts/fireworks/models/llama-v2-7b-chat")
    """

    model_id: str = Field(
        "accounts/fireworks/models/fireworks-llama-v2-7b-chat", alias="model"
    )
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    max_tokens: int = 512
    """The maximum number of tokens to generate in the completion.
    -1 returns as many tokens as possible given the prompt and
    the models maximal context size."""
    top_p: float = 1
    """Total probability mass of tokens to consider at each step."""
    fireworks_api_key: Optional[str] = None
    """Api key to use fireworks API"""
    batch_size: int = 20
    """Batch size to use when passing multiple documents to generate."""
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout for requests to Fireworks completion API. Default is 600 seconds."""

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"fireworks_api_key": "FIREWORKS_API_KEY"}

    @property
    def lc_serializable(self) -> bool:
        return True

    def __new__(cls, **data: Any) -> Any:
        """Initialize the Fireworks object."""
        model_id = data.get("model_id", "")
        return super().__new__(cls)

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["fireworks_api_key"] = get_from_dict_or_env(
            values, "fireworks_api_key", "FIREWORKS_API_KEY"
        )
        return values

    def completion_with_retry(
        llm: Union[BaseFireworks, FireworksChat], **kwargs: Any
    ) -> Any:
        """Use tenacity to retry the completion call."""
        answers = []
        result = execute(kwargs["messages"], kwargs["model"], llm.fireworks_api_key)
        print(result)
        curr_string = json.loads(result)["choices"][0]["message"]
        answers.append(curr_string)

        return answers

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        print(response)
        generations = []
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            gen = ChatGeneration(
                message=message,
                generation_info=dict(finish_reason=res.get("finish_reason")),
            )
            generations.append(gen)
        llm_output = {"token_usage": response["usage"], "model_name": self.model_name}
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call out to Fireworks endpoint with k unique prompts.
        Args:
            prompts: The prompts to pass into the model.
            stop: Optional list of stop words to use when generating.
        Returns:
            The full LLM output.
        """
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        params = {"model": self.model_id}
        params = {**params, **kwargs}
        response = self.completion_with_retry(messages=message_dicts, **params)
        return self._create_chat_result(response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call out to Fireworks endpoint async with k unique prompts."""
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        params = {"model": self.model_id}
        params = {**params, **kwargs}
        response = await acompletion_with_retry(self, messages=message_dicts, **params)
        return self._create_chat_result(response)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "fireworks-chat"
