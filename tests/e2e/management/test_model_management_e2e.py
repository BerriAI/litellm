"""Live e2e: the model-management routes' add / update / delete contract.

Each test provisions its own deployment through /model/new under a unique name
(deleted on teardown) and asserts both halves of the lifecycle contract: the
recorded state (/model/info reflects the write) and the enforced behavior (the
gateway serves or refuses traffic accordingly). Management writes land on the
control plane while chat is served by the data plane, which picks the change up
on its DB sync, so the traffic-facing read-backs poll to a deadline instead of
asserting once.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from e2e_config import unique_marker
from e2e_http import StreamingResponse, Success
from lifecycle import ResourceManager
from management_client import (
    ROUTE_NOT_ALLOWED_MARKER,
    ManagementClient,
    is_deleted_model_rejection,
)
from models import ChatBody, ChatMessage, ChatResponse, LiteLLMParamsBody, ModelInfoEntry

pytestmark = pytest.mark.e2e


def _chat_body(model: str) -> ChatBody:
    return ChatBody(
        model=model,
        messages=[ChatMessage(role="user", content=f"reply with one word {unique_marker()}")],
        max_tokens=128,
    )


def _poll[T](client: ManagementClient, attempt: Callable[[], T | None], failure: str) -> T:
    deadline = time.monotonic() + client.gateway.poll_timeout
    while time.monotonic() < deadline:
        found = attempt()
        if found is not None:
            return found
        time.sleep(client.gateway.poll_interval)
    pytest.fail(failure)


def _delete_if_present(client: ManagementClient, model_name: str) -> None:
    if client.find_deployment(model_name) is not None:
        client.gateway.delete_model(model_name)


def _provision(
    client: ManagementClient, resources: ResourceManager, *, tpm: int | None = None
) -> str:
    """Register a fresh gemini-backed deployment (id == name, cleaned up on
    teardown) and return its unique model name."""
    model_name = f"e2e-mgmt-{unique_marker()}"
    _ = client.gateway.create_model(
        model_name,
        LiteLLMParamsBody(model="gemini/gemini-3.5-flash", api_key="os.environ/GEMINI_API_KEY", tpm=tpm),
    )
    resources.defer(lambda: _delete_if_present(client, model_name))
    return model_name


def _poll_chat_ok(client: ManagementClient, key: str, model: str) -> ChatResponse:
    def attempt() -> ChatResponse | None:
        match client.gateway.chat(key, _chat_body(model)):
            case Success(data=data):
                return data
            case _:
                return None

    return _poll(
        client,
        attempt,
        f"{model} never became callable through the data plane before the deadline",
    )


def _poll_chat_rejects_deleted_model(client: ManagementClient, key: str, model: str) -> None:
    def attempt() -> StreamingResponse | None:
        outcome = client.chat_status(key, model, f"say hi {unique_marker()}")
        return outcome if is_deleted_model_rejection(outcome, model) else None

    _ = _poll(
        client,
        attempt,
        f"deleted model {model} was still served (never rejected with a 400 naming it) at the deadline",
    )


class TestModelManagementRoutes:
    @pytest.mark.covers("mgmt.model.add.persists")
    def test_add_persists_to_model_info_and_serves_chat(
        self, client: ManagementClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        model = _provision(client, resources, tpm=500000)

        entry = _poll(
            client,
            lambda: client.find_deployment(model),
            f"{model} never appeared in /model/info after /model/new",
        )
        assert entry.litellm_params.model == "gemini/gemini-3.5-flash", (
            f"/model/info reports backend {entry.litellm_params.model!r}, "
            f"configured 'gemini/gemini-3.5-flash'"
        )
        assert entry.litellm_params.tpm == 500000, (
            f"/model/info reports tpm {entry.litellm_params.tpm}, configured 500000"
        )

        response = _poll_chat_ok(client, scoped_key, model)
        assert response.choices, f"chat on {model} succeeded but returned no choices: {response}"

    @pytest.mark.covers("mgmt.model.update.persists")
    def test_update_tpm_persists_and_deployment_still_serves(
        self, client: ManagementClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        model = _provision(client, resources, tpm=500000)

        client.update_model_tpm(model, 424242)

        def updated() -> ModelInfoEntry | None:
            entry = client.find_deployment(model)
            if entry is not None and entry.litellm_params.tpm == 424242:
                return entry
            return None

        entry = _poll(
            client,
            updated,
            f"/model/info never reflected tpm 424242 for {model} after /model/update",
        )
        assert entry.litellm_params.model == "gemini/gemini-3.5-flash", (
            f"/model/update merge lost the backend model: {entry.litellm_params.model!r}"
        )

        response = _poll_chat_ok(client, scoped_key, model)
        assert response.choices, (
            f"updated deployment {model} no longer serves chat: {response}"
        )

    @pytest.mark.covers("mgmt.model.delete.persists")
    def test_delete_removes_from_model_info_and_rejects_chat(
        self, client: ManagementClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        model = _provision(client, resources)
        _ = _poll_chat_ok(client, scoped_key, model)

        client.delete_model_strict(model)

        def absent() -> bool | None:
            return True if client.find_deployment(model) is None else None

        _ = _poll(client, absent, f"{model} still listed in /model/info after /model/delete")
        _poll_chat_rejects_deleted_model(client, scoped_key, model)

    @pytest.mark.covers("mgmt.model.add.member_forbidden")
    def test_add_forbidden_for_llm_only_key(
        self, client: ManagementClient, resources: ResourceManager
    ) -> None:
        key = client.llm_only_key()
        resources.defer(lambda: client.gateway.delete_key(key))
        model = f"e2e-mgmt-forbidden-{unique_marker()}"

        outcome = client.create_model_status(
            key, model, LiteLLMParamsBody(model="gemini/gemini-3.5-flash")
        )

        assert outcome.status_code == 403, (
            f"llm-only key POSTing /model/new must be denied 403, got "
            f"{outcome.status_code}: {outcome.body[:300]}"
        )
        assert ROUTE_NOT_ALLOWED_MARKER in outcome.body, (
            f"403 body must be a route-permission denial, got: {outcome.body[:300]}"
        )
        assert client.find_deployment(model) is None, (
            f"{model} was created despite the 403 route denial"
        )
