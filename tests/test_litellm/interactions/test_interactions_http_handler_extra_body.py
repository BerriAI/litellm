"""VERIA-151 / VERIA-130: a client-supplied ``extra_body`` on the Interactions
API must not overwrite or inject the validated routing keys (model/agent) the
transform set, while genuine provider passthrough still merges through.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.interactions.http_handler import InteractionsHTTPHandler
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.gemini.interactions.transformation import (
    GoogleAIStudioInteractionsConfig,
)
from litellm.types.router import GenericLiteLLMParams


def _stub_response() -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.headers = {}
    response.json.return_value = {}
    response.text = "{}"
    return response


def _posted_json(is_async: bool, **call_kwargs) -> dict:
    """Run create/async_create_interaction with a stub client and return the
    JSON body that was actually POSTed (post extra_body merge)."""
    handler = InteractionsHTTPHandler()
    config = call_kwargs["interactions_api_config"]
    # transform_response is irrelevant here; stub it so we only exercise the
    # request build + extra_body merge and can read the posted body.
    with patch.object(config, "transform_response", return_value=MagicMock()):
        if is_async:
            client = MagicMock(spec=AsyncHTTPHandler)
            client.post = AsyncMock(return_value=_stub_response())
            asyncio.run(handler.async_create_interaction(client=client, **call_kwargs))
        else:
            client = MagicMock(spec=HTTPHandler)
            client.post = MagicMock(return_value=_stub_response())
            handler.create_interaction(client=client, **call_kwargs)
    return client.post.call_args.kwargs["json"]


def _base_kwargs():
    config = GoogleAIStudioInteractionsConfig()
    return {
        "interactions_api_config": config,
        "optional_params": {},
        "custom_llm_provider": "gemini",
        "litellm_params": GenericLiteLLMParams(api_key="AIza-test"),
        "logging_obj": MagicMock(),
        "model": "gemini/gemini-2.5-flash",
        "input": "hello",
    }


@pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
def test_extra_body_cannot_override_model(is_async):
    posted = _posted_json(
        is_async,
        **_base_kwargs(),
        extra_body={"model": "gemini/gemini-3-pro-ultra", "foo": "bar"},
    )
    # The validated (base) model is preserved, not the extra_body override.
    assert posted["model"] == "gemini-2.5-flash"
    # Legitimate passthrough still merges.
    assert posted["foo"] == "bar"


@pytest.mark.parametrize("is_async", [True, False], ids=["async", "sync"])
def test_extra_body_cannot_inject_agent(is_async):
    posted = _posted_json(
        is_async,
        **_base_kwargs(),
        extra_body={"agent": "deep-research-pro"},
    )
    # The request is model-routed; the unset routing key (agent) must not be
    # injectable through extra_body.
    assert posted["model"] == "gemini-2.5-flash"
    assert "agent" not in posted
