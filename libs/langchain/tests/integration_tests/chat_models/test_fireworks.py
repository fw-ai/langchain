"""Test ChatFireworks wrapper."""

import pytest

from langchain.callbacks.manager import CallbackManager
from langchain.chat_models.fireworks import ChatFireworks
from langchain.schema import (
    ChatGeneration,
    ChatResult,
    LLMResult,
)
from langchain.schema.messages import BaseMessage, HumanMessage, SystemMessage


def test_chat_fireworks() -> None:
    """Test ChatFireworks wrapper."""
    chat = ChatFireworks(
        model_id="accounts/fireworks/models/llama-v2-13b-chat", max_tokens=10
    )
    message = HumanMessage(content="Hello there")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_fireworks_model() -> None:
    """Test ChatFireworks wrapper handles model_name."""
    chat = ChatFireworks(model_id="foo")
    assert chat.model_id == "foo"
    chat = ChatFireworks(model_id="bar")
    assert chat.model_id == "bar"


def test_chat_fireworks_generate() -> None:
    """Test ChatFireworks wrapper with generate."""
    chat = ChatFireworks(max_tokens=10, n=2)
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 2
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


def test_chat_fireworks_multiple_completions() -> None:
    """Test ChatFireworks wrapper with multiple completions."""
    chat = ChatFireworks(max_tokens=10, n=5)
    message = HumanMessage(content="Hello")
    response = chat._generate([message])
    assert isinstance(response, ChatResult)
    assert len(response.generations) == 5
    for generation in response.generations:
        assert isinstance(generation.message, BaseMessage)
        assert isinstance(generation.message.content, str)


def test_chat_fireworks_llm_output_contains_model_id() -> None:
    """Test llm_output contains model_name."""
    chat = ChatFireworks(max_tokens=10)
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_id"] == chat.model_id


@pytest.mark.asyncio
async def test_async_chat_fireworks() -> None:
    """Test async generation."""
    chat = ChatFireworks(max_tokens=10, n=2)
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 2
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content
