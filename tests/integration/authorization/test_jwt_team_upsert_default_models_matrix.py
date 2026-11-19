import asyncio
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import anthropic
import httpx
import openai
import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, Scenario, eventually, gateway_from_environment
from tests.integration._support.database import read_rows
from tests.integration._support.scim import create_group, created_team
from tests.integration._support.wire import Reply, Request, Wire, wire_server
from tests.integration.authorization._jwt_team_upsert import (
    UpsertRig,
    delete_team_if_present,
    marker_of,
    prompt_of,
    provider_reply,
    team_models,
    upserting_jwt_rig,
)

pytestmark: Final = pytest.mark.shares_worker_state

PROVIDER_OUTAGE: Final = "scripted provider outage"
BURST: Final = 20


@dataclass(frozen=True, slots=True)
class Matrix:
    rig: UpsertRig
    scenario: Scenario
    provider: Wire
    approved: str
    unapproved: str
    subject: str

    def new_team(self) -> str:
        team_id: Final = f"integration-dept-{uuid.uuid4().hex}"
        self.scenario.cleanups.callback(delete_team_if_present, self.scenario, team_id)
        return team_id

    def token(self, team_id: JsonValue) -> str:
        return self.rig.sign({"sub": self.subject, "dept": team_id})

    def base_url(self) -> str:
        return str(self.rig.gateway.client.base_url).rstrip("/")

    def provider_markers(self) -> frozenset[str]:
        return frozenset(marker_of(request) for request in self.provider.drain())


def _provider(request: Request) -> Reply:
    if PROVIDER_OUTAGE in prompt_of(request):
        return Reply(status=500, body=b'{"error": {"message": "scripted provider outage", "type": "server_error"}}')
    return provider_reply(request)


@pytest.fixture(scope="module")
def shared_matrix(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Matrix]:
    with gateway_from_environment() as gateway, wire_server(_provider) as provider, gateway.scenario() as scenario:
        approved: Final = scenario.model(api_base=f"{provider.url}/v1")
        unapproved: Final = scenario.model(api_base=f"{provider.url}/v1")
        subject: Final = scenario.user(user_role="internal_user")
        with upserting_jwt_rig(
            gateway,
            tmp_path_factory.mktemp("jwt-team-upsert-matrix"),
            {"team_id_jwt_field": "dept"},
            {"models": [approved]},
            workers=2,
        ) as rig:
            yield Matrix(rig, scenario, provider, approved, unapproved, subject)


@pytest.fixture
def matrix(shared_matrix: Matrix) -> Matrix:
    shared_matrix.provider.drain()
    return shared_matrix


def _marker() -> str:
    return f"marker-{uuid.uuid4().hex}"


Call = Callable[[Matrix, str, str, str], tuple[int, str]]


def _chat_raw(matrix: Matrix, token: str, model: str, text: str) -> tuple[int, str]:
    response: Final = matrix.rig.gateway.request(
        "POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": text}]}, key=token
    )
    return response.status_code, response.text


def _chat_sdk(matrix: Matrix, token: str, model: str, text: str) -> tuple[int, str]:
    with openai.OpenAI(base_url=f"{matrix.base_url()}/v1", api_key=token, max_retries=0) as client:
        try:
            completion: Final = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": text}]
            )
        except openai.APIStatusError as error:
            return error.status_code, error.message
    return 200, f"{completion.id} {completion.choices[0].message.content}"


def _chat_stream_async_sdk(matrix: Matrix, token: str, model: str, text: str) -> tuple[int, str]:
    async def consume() -> tuple[int, str]:
        async with openai.AsyncOpenAI(base_url=f"{matrix.base_url()}/v1", api_key=token, max_retries=0) as client:
            try:
                stream: Final = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": text}], stream=True
                )
                chunks: Final = [chunk async for chunk in stream]
            except openai.APIStatusError as error:
                return error.status_code, error.message
        first_choices: Final = tuple(chunk.choices[0] for chunk in chunks if chunk.choices)
        return 200, " ".join((chunks[0].id, *(choice.delta.content or "" for choice in first_choices)))

    return asyncio.run(consume())


def _messages_sdk(matrix: Matrix, token: str, model: str, text: str) -> tuple[int, str]:
    with anthropic.Anthropic(base_url=matrix.base_url(), auth_token=token, max_retries=0) as client:
        try:
            message: Final = client.messages.create(
                model=model, max_tokens=16, messages=[{"role": "user", "content": text}]
            )
        except anthropic.APIStatusError as error:
            return error.status_code, error.message
    return 200, " ".join(block.text for block in message.content if block.type == "text")


def _messages_stream_async_sdk(matrix: Matrix, token: str, model: str, text: str) -> tuple[int, str]:
    async def consume() -> tuple[int, str]:
        async with anthropic.AsyncAnthropic(base_url=matrix.base_url(), auth_token=token, max_retries=0) as client:
            try:
                async with client.messages.stream(
                    model=model, max_tokens=16, messages=[{"role": "user", "content": text}]
                ) as stream:
                    return 200, await stream.get_final_text()
            except anthropic.APIStatusError as error:
                return error.status_code, error.message

    return asyncio.run(consume())


def _responses_sdk(matrix: Matrix, token: str, model: str, text: str) -> tuple[int, str]:
    with openai.OpenAI(base_url=f"{matrix.base_url()}/v1", api_key=token, max_retries=0) as client:
        try:
            response: Final = client.responses.create(model=model, input=text)
        except openai.APIStatusError as error:
            return error.status_code, error.message
    return 200, response.output_text


def _responses_stream_async_sdk(matrix: Matrix, token: str, model: str, text: str) -> tuple[int, str]:
    async def consume() -> tuple[int, str]:
        async with openai.AsyncOpenAI(base_url=f"{matrix.base_url()}/v1", api_key=token, max_retries=0) as client:
            try:
                stream: Final = await client.responses.create(model=model, input=text, stream=True)
                events: Final = [event async for event in stream]
            except openai.APIStatusError as error:
                return error.status_code, error.message
        return 200, "".join(event.delta for event in events if event.type == "response.output_text.delta")

    return asyncio.run(consume())


LLM_CALLS: Final[Mapping[str, Call]] = {
    "chat-raw-httpx": _chat_raw,
    "chat-openai-sdk": _chat_sdk,
    "chat-stream-async-openai-sdk": _chat_stream_async_sdk,
    "messages-anthropic-sdk": _messages_sdk,
    "messages-stream-async-anthropic-sdk": _messages_stream_async_sdk,
    "responses-openai-sdk": _responses_sdk,
    "responses-stream-async-openai-sdk": _responses_stream_async_sdk,
}


def _served_spend_rows(team_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT model_group FROM "LiteLLM_SpendLogs" WHERE team_id = %s AND status = %s', (team_id, "success")
    )


@pytest.mark.parametrize("call", LLM_CALLS.values(), ids=LLM_CALLS.keys())
def test_first_jwt_request_on_every_llm_route_upserts_a_team_limited_to_the_default_models(
    matrix: Matrix, call: Call
) -> None:
    team_id: Final = matrix.new_team()
    served_marker: Final = _marker()
    refused_marker: Final = _marker()

    served: Final = call(matrix, matrix.token(team_id), matrix.approved, served_marker)
    refused: Final = call(matrix, matrix.token(team_id), matrix.unapproved, refused_marker)

    assert (served[0], served_marker in served[1], refused[0]) == (200, True, 403), (served, refused)
    assert matrix.provider_markers() == {served_marker}
    assert team_models(team_id) == [[matrix.approved]]
    spend: Final = eventually(lambda: _served_spend_rows(team_id), lambda rows: len(rows) >= 1, seconds=70)
    assert spend == [{"model_group": matrix.approved}]


def test_a_jwt_request_on_a_non_llm_route_upserts_the_team_with_the_default_models(matrix: Matrix) -> None:
    team_id: Final = matrix.new_team()

    listed: Final = matrix.rig.gateway.request("GET", "/v1/models", key=matrix.token(team_id))

    assert listed.status_code == 200, listed.text
    assert [entry["id"] for entry in listed.json()["data"]] == [matrix.approved], listed.text
    assert team_models(team_id) == [[matrix.approved]]


def test_a_provider_failure_on_the_upserting_request_still_leaves_the_team_on_the_default_models(
    matrix: Matrix,
) -> None:
    team_id: Final = matrix.new_team()
    token: Final = matrix.token(team_id)

    failed: Final = _chat_raw(matrix, token, matrix.approved, f"{PROVIDER_OUTAGE} {_marker()}")
    recovered_marker: Final = _marker()
    recovered: Final = _chat_raw(matrix, token, matrix.approved, recovered_marker)

    assert (failed[0], PROVIDER_OUTAGE in failed[1]) == (500, True), failed
    assert (recovered[0], recovered_marker in recovered[1]) == (200, True), recovered
    assert team_models(team_id) == [[matrix.approved]]


@pytest.mark.parametrize("credential", ["forged-signature", "no-credential"])
def test_an_unauthenticated_request_naming_a_new_team_is_refused_and_creates_no_team(
    matrix: Matrix, credential: str
) -> None:
    team_id: Final = matrix.new_team()
    marker: Final = _marker()
    headers: Final = (
        {"Authorization": f"Bearer {matrix.rig.forge({'sub': matrix.subject, 'dept': team_id})}"}
        if credential == "forged-signature"
        else {}
    )

    with httpx.Client(base_url=matrix.base_url(), timeout=30, trust_env=False) as anonymous:
        refused: Final = anonymous.post(
            "/v1/chat/completions",
            json={"model": matrix.approved, "messages": [{"role": "user", "content": marker}]},
            headers=headers,
        )
    control: Final = _chat_raw(matrix, matrix.token(matrix.new_team()), matrix.approved, _marker())

    assert (refused.status_code, control[0]) == (401, 200), (refused.text, control)
    assert marker not in matrix.provider_markers()
    assert team_models(team_id) == []


def test_repeating_the_upserting_request_keeps_one_team_row_with_the_default_models(matrix: Matrix) -> None:
    team_id: Final = matrix.new_team()
    token: Final = matrix.token(team_id)

    def created_at() -> list[dict[str, JsonValue]]:
        return read_rows(
            'SELECT created_at::text AS created_at, models FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team_id,)
        )

    first: Final = _chat_raw(matrix, token, matrix.approved, _marker())
    after_first: Final = created_at()
    second: Final = _chat_raw(matrix, token, matrix.approved, _marker())

    assert (first[0], second[0]) == (200, 200), (first, second)
    assert [row["models"] for row in after_first] == [[matrix.approved]]
    assert created_at() == after_first


def test_models_an_admin_sets_on_an_upserted_team_are_not_reset_to_the_defaults_by_later_jwt_requests(
    matrix: Matrix,
) -> None:
    team_id: Final = matrix.new_team()
    token: Final = matrix.token(team_id)
    upserted: Final = _chat_raw(matrix, token, matrix.approved, _marker())
    assert upserted[0] == 200, upserted

    updated: Final = matrix.rig.gateway.request(
        "POST", "/team/update", {"team_id": team_id, "models": [matrix.unapproved]}
    )
    assert updated.status_code == 200, updated.text
    now_served: Final = eventually(
        lambda: _chat_raw(matrix, token, matrix.unapproved, _marker()), lambda answer: answer[0] == 200, seconds=70
    )
    now_refused: Final = eventually(
        lambda: _chat_raw(matrix, token, matrix.approved, _marker()), lambda answer: answer[0] == 403, seconds=70
    )

    assert (now_served[0], now_refused[0]) == (200, 403), (now_served, now_refused)
    assert team_models(team_id) == [[matrix.unapproved]]


def test_team_update_of_an_admin_created_team_leaves_its_models_empty(matrix: Matrix) -> None:
    team_id: Final = matrix.new_team()
    created: Final = matrix.rig.gateway.request("POST", "/team/new", {"team_id": team_id})
    assert created.status_code == 200, created.text

    updated: Final = matrix.rig.gateway.request(
        "POST", "/team/update", {"team_id": team_id, "team_alias": f"renamed-{team_id}"}
    )

    assert updated.status_code == 200, updated.text
    assert team_models(team_id) == [[]]


def test_a_scim_group_team_takes_the_default_team_models(matrix: Matrix) -> None:
    team_id: Final = created_team(matrix.scenario, create_group(matrix.rig.gateway, ()))

    assert team_models(team_id) == [[matrix.approved]]


def test_concurrent_first_requests_for_one_new_team_create_it_once_with_the_default_models(matrix: Matrix) -> None:
    team_id: Final = matrix.new_team()
    token: Final = matrix.token(team_id)
    markers: Final = tuple(_marker() for _ in range(BURST))

    with ThreadPoolExecutor(max_workers=BURST) as pool:
        answers: Final = tuple(pool.map(lambda marker: _chat_raw(matrix, token, matrix.approved, marker), markers))
    served: Final = frozenset(marker for marker, (status, _) in zip(markers, answers, strict=True) if status == 200)
    lost_the_creation_race: Final = tuple(
        (status, text) for status, text in answers if status != 200 and (status != 404 or team_id not in text)
    )
    settled: Final = _chat_raw(matrix, token, matrix.approved, _marker())
    refused: Final = _chat_raw(matrix, token, matrix.unapproved, _marker())

    assert (len(served) >= 1, lost_the_creation_race) == (True, ()), answers
    assert served <= matrix.provider_markers()
    assert (team_models(team_id), settled[0], refused[0]) == ([[matrix.approved]], 200, 403), (settled, refused)


@pytest.mark.parametrize(
    ("claim", "expected_status", "team_created"),
    [(314159, 403, True), ("", 401, False), ("x" * 5120, 403, True), (["integration-dept-in-a-list-"], 403, True)],
    ids=["int", "empty-string", "five-kb-string", "list-in-scalar-claim"],
)
def test_a_team_upserted_from_an_unusual_team_claim_is_still_limited_to_the_default_models(
    matrix: Matrix, claim: JsonValue, expected_status: int, team_created: bool
) -> None:
    suffix: Final = uuid.uuid4().hex
    unique: Final[JsonValue] = (
        [f"{claim[0]}{suffix}"]
        if isinstance(claim, list)
        else f"{claim}{suffix}"
        if isinstance(claim, str) and claim
        else uuid.uuid4().int >> 76
        if isinstance(claim, int)
        else claim
    )
    named: Final = str(unique[0] if isinstance(unique, list) else unique)
    matrix.scenario.cleanups.callback(delete_team_if_present, matrix.scenario, named)
    marker: Final = _marker()
    existing: Final = team_models(named)

    answer: Final = _chat_raw(matrix, matrix.token(unique), matrix.unapproved, marker)
    control: Final = _chat_raw(matrix, matrix.token(matrix.new_team()), matrix.approved, _marker())

    assert control[0] == 200, control
    assert existing == [], named
    assert (answer[0], team_models(named)) == (expected_status, [[matrix.approved]] if team_created else []), answer
    assert marker not in matrix.provider_markers()


def _control_chat(rig: UpsertRig, model: str, key: str | None) -> httpx.Response:
    return rig.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": "team upsert control"}]},
        key=key,
    )


@pytest.mark.parametrize(
    "defaults",
    [None, {"models": []}, {"models": None}, {"team_member_permissions": ["/key/info"]}],
    ids=["no-default-team-params", "empty-models", "null-models", "defaults-without-models"],
)
def test_a_team_upserted_without_configured_default_models_starts_with_no_model_restriction(
    gateway: Gateway, tmp_path: Path, defaults: Mapping[str, JsonValue] | None
) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = scenario.user(user_role="internal_user")
        team_id: Final = f"integration-dept-{uuid.uuid4().hex}"
        with upserting_jwt_rig(gateway, tmp_path, {"team_id_jwt_field": "dept"}, defaults) as rig:
            scenario.cleanups.callback(delete_team_if_present, scenario, team_id)

            served: Final = _control_chat(rig, model, rig.sign({"sub": subject, "dept": team_id}))

            assert (served.status_code, team_models(team_id)) == (200, [[]]), served.text


def test_duplicated_default_models_are_stored_as_configured_on_the_upserted_team(
    gateway: Gateway, tmp_path: Path
) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = scenario.user(user_role="internal_user")
        team_id: Final = f"integration-dept-{uuid.uuid4().hex}"
        with upserting_jwt_rig(gateway, tmp_path, {"team_id_jwt_field": "dept"}, {"models": [model, model]}) as rig:
            scenario.cleanups.callback(delete_team_if_present, scenario, team_id)

            served: Final = _control_chat(rig, model, rig.sign({"sub": subject, "dept": team_id}))

            assert (served.status_code, team_models(team_id)) == (200, [[model, model]]), served.text


@pytest.mark.parametrize("shape", ["string", "integers"])
def test_malformed_default_models_refuse_the_upsert_without_creating_a_team(
    gateway: Gateway, tmp_path: Path, shape: str
) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = scenario.user(user_role="internal_user")
        team_id: Final = f"integration-dept-{uuid.uuid4().hex}"
        models: Final[JsonValue] = model if shape == "string" else [1, 2]
        with upserting_jwt_rig(gateway, tmp_path, {"team_id_jwt_field": "dept"}, {"models": models}) as rig:
            scenario.cleanups.callback(delete_team_if_present, scenario, team_id)

            refused: Final = _control_chat(rig, model, rig.sign({"sub": subject, "dept": team_id}))
            admin: Final = _control_chat(rig, model, None)

            assert (refused.status_code, team_models(team_id), admin.status_code) == (404, [], 200), (
                refused.text,
                admin.text,
            )
