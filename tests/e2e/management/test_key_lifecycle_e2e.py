"""Live e2e: one virtual key walked through its whole lifecycle, read back on every
gateway replica.

Create, read, partial update, clear, enforce, delete: one method per step, and every
step creates its own team and key (both deleted on teardown) so a step reruns or skips
on its own. Writes go through the control plane; read-backs poll every URL in
PROXY_REPLICA_URLS until each replica converges, because a write that is visible on the
gateway that took it and stale on its neighbour is exactly the failure this file exists
to catch. Revocation is the slowest of those: a deleted key stays usable on the other
replicas until their auth cache entry expires, so the delete step polls each of them
rather than asserting once.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import pytest

from e2e_config import unique_marker
from e2e_http import Result, StreamingResponse, Success, UnknownApiError, unwrap
from lifecycle import ResourceManager
from management_client import MODEL_ACCESS_DENIED_MARKER, ManagementClient
from models import (
    CLEAR,
    ChatBody,
    ChatMessage,
    KeyGenerateBody,
    KeyGenerateResponse,
    KeyInfo,
    KeyInfoParams,
    KeyInfoResponse,
    KeyMetadata,
    KeyUpdateBody,
    LiteLLMParamsBody,
    TeamNewBody,
)
from proxy_client import Converged, NotConverged, Poller, await_converged, await_converged_everywhere
from transport import Transport

pytestmark = pytest.mark.e2e

BACKING_MODEL: Final = "gpt-4o-mini"
DENIED_MODEL: Final = "gpt-5.5"
MAX_BUDGET: Final = 25.0
TPM_LIMIT: Final = 313131
RPM_LIMIT: Final = 323232
UPDATED_RPM_LIMIT: Final = 424242
BUDGET_DURATION: Final = "30d"


@dataclass(frozen=True, slots=True)
class CreatedKey:
    written: KeyGenerateBody
    response: KeyGenerateResponse

    @property
    def key(self) -> str:
        return self.response.key


@pytest.fixture(scope="module")
def mock_deployment(client: ManagementClient) -> Iterator[str]:
    """A deployment that answers from a canned response, so the enforcement step needs no
    provider key. The alias carries a unique marker, like every other model this suite
    registers, so concurrent runs never share one model group."""
    model_name: Final = f"e2e-key-lifecycle-{unique_marker()}"
    model_id: Final = client.proxy.create_model(model_name, LiteLLMParamsBody(model=BACKING_MODEL, mock_response="ok"))
    try:
        yield model_name
    finally:
        client.proxy.delete_model(model_id)


def _await[T](client: ManagementClient, poller: Poller[T], converged: Callable[[T], bool], failure: str) -> T:
    outcome: Final = await_converged(
        poller,
        converged=converged,
        timeout=client.proxy.poll_timeout,
        interval=client.proxy.poll_interval,
        now=time.monotonic,
        sleep=time.sleep,
    )
    match outcome:
        case Converged(result=result):
            return result
        case NotConverged(last_result=last):
            pytest.fail(f"{failure}; last outcome: {last}")


def _chat_poller(transport: Transport, key: str, model: str) -> Poller[StreamingResponse]:
    return lambda: transport.send(
        "/chat/completions",
        headers=transport.bearer(key),
        json=ChatBody(
            model=model,
            messages=[ChatMessage(role="user", content=f"say hi {unique_marker()}")],
            max_tokens=16,
        ),
    )


def _create_key(client: ManagementClient, resources: ResourceManager, model: str) -> CreatedKey:
    marker: Final = unique_marker()
    team_id: Final = client.create_team(TeamNewBody(team_alias=f"e2e-key-lifecycle-team-{marker}"))
    resources.defer(lambda: client.delete_team(team_id))
    written: Final = KeyGenerateBody(
        key_alias=f"e2e-key-lifecycle-{marker}",
        models=[model],
        max_budget=MAX_BUDGET,
        tpm_limit=TPM_LIMIT,
        rpm_limit=RPM_LIMIT,
        budget_duration=BUDGET_DURATION,
        metadata=KeyMetadata(tag=marker),
        team_id=team_id,
    )
    response: Final = unwrap(client.generate_key(written))
    resources.defer(lambda: client.proxy.delete_key(response.key))
    return CreatedKey(written=written, response=response)


def _key_info_everywhere(
    client: ManagementClient, key: str, settled: Callable[[KeyInfo], bool]
) -> Mapping[str, KeyInfo]:
    def converged(result: Result[KeyInfoResponse]) -> bool:
        return isinstance(result, Success) and settled(result.data.info)

    reads: Final = client.proxy.read_back_everywhere(
        "/key/info", params=KeyInfoParams(key=key), response_type=KeyInfoResponse, converged=converged
    )
    return MappingProxyType({replica: unwrap(read).info for replica, read in reads.items()})


def _is_key_not_found(result: Result[KeyInfoResponse]) -> bool:
    return isinstance(result, UnknownApiError) and result.status_code == 404


def _assert_reads_back(info: KeyInfo, expected: KeyGenerateBody, replica: str) -> None:
    for field, observed, wanted in (
        ("key_alias", info.key_alias, expected.key_alias),
        ("models", info.models, expected.models),
        ("max_budget", info.max_budget, expected.max_budget),
        ("tpm_limit", info.tpm_limit, expected.tpm_limit),
        ("rpm_limit", info.rpm_limit, expected.rpm_limit),
        ("budget_duration", info.budget_duration, expected.budget_duration),
        ("team_id", info.team_id, expected.team_id),
        ("metadata", info.metadata, expected.metadata),
    ):
        assert observed == wanted, f"{replica}: /key/info reports {field}={observed!r}, expected {wanted!r}"


def _poll_chat_ok(client: ManagementClient, key: str, model: str) -> None:
    _ = _await(
        client,
        _chat_poller(client.proxy.transport, key, model),
        lambda outcome: outcome.ok,
        f"chat on {model} never succeeded for the key before the deadline",
    )


def _warm_every_replica(client: ManagementClient, key: str, model: str) -> None:
    """Serve one call from every replica, so each has the key in its auth cache. Without
    this the revocation check below would only prove a replica rejects a key it never
    knew, which is true of any random string."""
    for replica, transport in client.proxy.replicas.items():
        _ = _await(
            client,
            _chat_poller(transport, key, model),
            lambda outcome: outcome.ok,
            f"{replica}: chat on {model} never succeeded for the key before the deadline",
        )


def _assert_chat_rejected_everywhere(client: ManagementClient, key: str, model: str) -> None:
    outcomes: Final = await_converged_everywhere(
        {replica: _chat_poller(transport, key, model) for replica, transport in client.proxy.replicas.items()},
        converged=lambda outcome: outcome.status_code == 401,
        timeout=client.proxy.poll_timeout,
        interval=client.proxy.poll_interval,
        now=time.monotonic,
        sleep=time.sleep,
    )
    for replica, outcome in outcomes.items():
        assert isinstance(outcome, Converged), (
            f"{replica}: the deleted key was still accepted on chat after "
            f"{client.proxy.poll_timeout}s, last status {outcome.last_result.status_code}"
        )


class TestKeyLifecycle:
    def test_create_echoes_every_field_written(
        self, client: ManagementClient, resources: ResourceManager, mock_deployment: str
    ) -> None:
        created: Final = _create_key(client, resources, mock_deployment)

        response: Final = created.response
        for field, observed, wanted in (
            ("key_alias", response.key_alias, created.written.key_alias),
            ("models", response.models, created.written.models),
            ("max_budget", response.max_budget, created.written.max_budget),
            ("tpm_limit", response.tpm_limit, created.written.tpm_limit),
            ("rpm_limit", response.rpm_limit, created.written.rpm_limit),
            ("budget_duration", response.budget_duration, created.written.budget_duration),
            ("team_id", response.team_id, created.written.team_id),
            ("metadata", response.metadata, created.written.metadata),
        ):
            assert observed == wanted, f"/key/generate echoed {field}={observed!r}, sent {wanted!r}"

    def test_read_reflects_the_create_on_every_replica(
        self, client: ManagementClient, resources: ResourceManager, mock_deployment: str
    ) -> None:
        created: Final = _create_key(client, resources, mock_deployment)

        infos: Final = _key_info_everywhere(
            client, created.key, lambda info: info.key_alias == created.written.key_alias
        )
        for replica, info in infos.items():
            _assert_reads_back(info, created.written, replica)
            assert info.budget_reset_at is not None, (
                f"{replica}: /key/info reports no budget_reset_at for budget_duration={BUDGET_DURATION!r}"
            )

    @pytest.mark.covers("mgmt.key.update.preserves_unrelated_fields")
    def test_partial_update_changes_only_the_named_field(
        self, client: ManagementClient, resources: ResourceManager, mock_deployment: str
    ) -> None:
        created: Final = _create_key(client, resources, mock_deployment)
        before: Final = _key_info_everywhere(client, created.key, lambda info: info.rpm_limit == RPM_LIMIT)

        _ = unwrap(client.update_key(KeyUpdateBody(key=created.key, rpm_limit=UPDATED_RPM_LIMIT)))

        after: Final = _key_info_everywhere(client, created.key, lambda info: info.rpm_limit == UPDATED_RPM_LIMIT)
        for replica, info in after.items():
            _assert_reads_back(info, created.written.model_copy(update={"rpm_limit": UPDATED_RPM_LIMIT}), replica)
            assert info.budget_reset_at == before[replica].budget_reset_at, (
                f"{replica}: budget_reset_at moved from {before[replica].budget_reset_at!r} to "
                f"{info.budget_reset_at!r} on a /key/update that did not name budget_duration"
            )

    @pytest.mark.covers("mgmt.key.update.clear_persists")
    def test_explicit_null_clears_the_budget_and_its_reset_time(
        self, client: ManagementClient, resources: ResourceManager, mock_deployment: str
    ) -> None:
        created: Final = _create_key(client, resources, mock_deployment)
        _ = _key_info_everywhere(client, created.key, lambda info: info.max_budget == MAX_BUDGET)

        _ = unwrap(client.update_key(KeyUpdateBody(key=created.key, max_budget=CLEAR, budget_duration=CLEAR)))

        cleared: Final = _key_info_everywhere(client, created.key, lambda info: info.max_budget is None)
        for replica, info in cleared.items():
            assert info.budget_duration is None, (
                f"{replica}: budget_duration={info.budget_duration!r} survived an explicit null"
            )
            assert info.budget_reset_at is None, (
                f"{replica}: clearing budget_duration left budget_reset_at={info.budget_reset_at!r}"
            )
            _assert_reads_back(
                info, created.written.model_copy(update={"max_budget": None, "budget_duration": None}), replica
            )

    def test_key_serves_its_model_and_is_denied_others(
        self, client: ManagementClient, resources: ResourceManager, mock_deployment: str
    ) -> None:
        created: Final = _create_key(client, resources, mock_deployment)

        _poll_chat_ok(client, created.key, mock_deployment)

        denied: Final = client.chat_status(created.key, DENIED_MODEL, f"say hi {unique_marker()}")
        assert denied.status_code == 403, (
            f"chat on {DENIED_MODEL!r} outside the key's model list must be denied 403, got "
            f"{denied.status_code}: {denied.body[:300]}"
        )
        assert MODEL_ACCESS_DENIED_MARKER in denied.body, (
            f"403 body must be a model-access denial, got: {denied.body[:300]}"
        )

    def test_delete_revokes_info_and_chat_on_every_replica(
        self, client: ManagementClient, resources: ResourceManager, mock_deployment: str
    ) -> None:
        """The teardown's deferred delete fires again on the already-deleted key by
        design: the deferred cleanup must survive this test failing before the
        in-body delete, and a repeat /key/delete is a cheap no-op the warn-only
        teardown absorbs."""
        created: Final = _create_key(client, resources, mock_deployment)
        _warm_every_replica(client, created.key, mock_deployment)

        client.delete_key_strict(created.key)

        _ = client.proxy.read_back_everywhere(
            "/key/info",
            params=KeyInfoParams(key=created.key),
            response_type=KeyInfoResponse,
            converged=_is_key_not_found,
        )
        _assert_chat_rejected_everywhere(client, created.key, mock_deployment)
