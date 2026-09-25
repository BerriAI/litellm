"""Live e2e: an Azure code_interpreter container's file, read back by its native
id with a team service-account key.

The Azure container endpoints were first verified on one shape: a single Azure
deployment whose credentials came from ``AZURE_API_BASE`` in the proxy env,
containers created explicitly with LiteLLM-managed ids, and the master key as
the caller. The customer differs on all three axes at once, and this cell pins
that shape:

- the Azure deployments carry their own ``api_base`` and ``api_key`` (the
  pytest process reads both from its env and registers them literally), so a
  proxy booted with no ``AZURE_API_BASE`` serves them;
- two Azure deployments are registered, the first with an invalid key, so a
  container call that guesses a deployment instead of routing by container id
  lands on the decoy and fails;
- the container is created implicitly by ``/v1/responses`` with the
  ``code_interpreter`` tool, and every container call afterwards names it by
  Azure's own ``cntr_<hex>`` id with only ``custom_llm_provider=azure`` beside
  it, the way a client that stores provider ids does (the routing envelope
  LiteLLM wraps around the id in the responses output is peeled off first);
- every LLM-side call is made with a service-account key of a team whose
  member is a plain ``internal_user``; a service-account key belongs to the
  team, not to a user, and the master key only does the setup.

Fail-before-fix, proven against a local proxy booted with no Azure env: with
#28990 reverted the upload 403s (the ownership row written at creation no
longer matches a key without a user_id), and with #27921 reverted it fails
with "api_base is required for Azure AI Studio ... Passed `api_base=None`"
because the native id carries no model_id and nothing else names a deployment.
A proxy whose env carries ``AZURE_API_BASE`` for the same resource masks the
second regression, since the global-credential fallback then reaches the
container anyway.

The streaming variant is not here: a streamed ``/v1/responses`` writes the
container ownership row only after the ``[DONE]`` frame, and the OpenAI SDK
closes the connection at ``[DONE]``, so the write is cancelled and every
follow-up container call 403s (LIT-8612). That cell comes with its fix.
"""

from __future__ import annotations

import base64
import binascii
import os
import re
from types import MappingProxyType
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from management.management_client import ManagementClient, build_client
from models import KeyGenerateBody, KeyGenerateResponse, LiteLLMParamsBody, TeamNewBody, UserNewBody
from openai import OpenAI
from openai.types.responses import Response, ResponseCodeInterpreterToolCall
from openai.types.responses.tool_param import CodeInterpreter
from proxy_client import ProxyClient
from sdk_clients import NO_PROXY_CACHE, SdkClients

pytestmark = [pytest.mark.e2e, pytest.mark.provider_live]

AZURE_BACKEND: Final = "azure/gpt-5.4-nano"
AZURE_API_VERSION: Final = "v1"
AZURE_PROVIDER_QUERY: Final = MappingProxyType({"custom_llm_provider": "azure"})
CODE_INTERPRETER: Final[CodeInterpreter] = {"type": "code_interpreter", "container": {"type": "auto"}}
PROMPT: Final = "Use python to compute 6*7 and reply with just the number."
NATIVE_CONTAINER_ID: Final = re.compile(r"cntr_[0-9a-f]+")


def _azure_params(api_base: str, api_key: str) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(model=AZURE_BACKEND, api_base=api_base, api_key=api_key, api_version=AZURE_API_VERSION)


def _register_two_azure_deployments(proxy: ProxyClient, resources: ResourceManager, marker: str) -> str:
    api_base: Final = os.environ.get("AZURE_API_BASE")
    api_key: Final = os.environ.get("AZURE_API_KEY")
    if not api_base or not api_key:
        pytest.skip("set AZURE_API_BASE and AZURE_API_KEY in the pytest env; the deployments are registered with them")
    decoy_id: Final = proxy.create_model(
        f"e2e-containers-decoy-{marker}", _azure_params(api_base, f"decoy-{marker}"), provider_live=True
    )
    resources.defer(lambda: proxy.delete_model(decoy_id))
    model: Final = f"e2e-containers-{marker}"
    model_id: Final = proxy.create_model(model, _azure_params(api_base, api_key), provider_live=True)
    resources.defer(lambda: proxy.delete_model(model_id))
    return model


def _service_account_key(
    proxy: ProxyClient, resources: ResourceManager, management: ManagementClient, marker: str, model: str
) -> str:
    team_id: Final = management.create_team(TeamNewBody(team_alias=f"e2e-containers-{marker}", models=[model]))
    resources.defer(lambda: management.delete_team(team_id))
    user_id: Final = management.create_user(
        UserNewBody(user_email=f"e2e-containers-{marker}@example.com", user_role="internal_user")
    )
    resources.defer(lambda: management.delete_user_strict(user_id))
    management.add_team_member(team_id, user_id)
    resources.defer(lambda: management.delete_team_member(team_id, user_id))
    generated: Final = unwrap(
        proxy.transport.post(
            "/key/service-account/generate",
            headers=proxy.management_headers(),
            json=KeyGenerateBody(team_id=team_id, key_alias=f"e2e-containers-sa-{marker}", models=[model]),
            response_type=KeyGenerateResponse,
        )
    )
    resources.defer(lambda: management.delete_key_strict(generated.key))
    return generated.key


def _response_with_code_interpreter(client: OpenAI, model: str) -> Response:
    return client.responses.create(model=model, input=PROMPT, tools=[CODE_INTERPRETER], extra_body=NO_PROXY_CACHE)


def _container_id(response: Response) -> str:
    calls: Final = tuple(item for item in response.output if isinstance(item, ResponseCodeInterpreterToolCall))
    assert calls, f"no code_interpreter_call in the responses output: {response.output!r}"
    return calls[0].container_id


def _native_container_id(container_id: str) -> str:
    try:
        envelope: Final = base64.b64decode(container_id.removeprefix("cntr_"), validate=True).decode()
    except (binascii.Error, UnicodeDecodeError):
        return container_id
    return envelope.rpartition("container_id:")[2] if envelope.startswith("litellm:") else container_id


def _assert_file_round_trip(client: OpenAI, native_id: str, marker: str) -> None:
    payload: Final = f"hello from {marker}\n".encode()
    uploaded: Final = client.containers.files.create(
        native_id, file=(f"{marker}.txt", payload), extra_query=AZURE_PROVIDER_QUERY
    )
    fetched: Final = client.containers.files.content.retrieve(
        uploaded.id, container_id=native_id, extra_query=AZURE_PROVIDER_QUERY
    )
    assert fetched.content == payload, f"container file content differs from the upload: {fetched.content!r}"


class TestAzureContainerFiles:
    @pytest.mark.covers("llm.responses.azure_openai.code_interpreter.nonstream.works")
    def test_service_account_key_reads_container_file_by_native_id(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        marker: Final = unique_marker()
        model: Final = _register_two_azure_deployments(proxy, resources, marker)
        key: Final = _service_account_key(proxy, resources, build_client(proxy), marker, model)
        client: Final = sdk.openai(key)
        native_id: Final = _native_container_id(_container_id(_response_with_code_interpreter(client, model)))
        assert NATIVE_CONTAINER_ID.fullmatch(native_id), f"container id is not a native Azure id: {native_id}"
        resources.defer(lambda: client.containers.delete(native_id, extra_query=AZURE_PROVIDER_QUERY))
        _assert_file_round_trip(client, native_id, marker)
