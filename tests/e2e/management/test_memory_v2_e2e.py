import hashlib
import os
from dataclasses import dataclass
from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_http import Success, unwrap
from lifecycle import ResourceManager
from management_client import ManagementClient
from memory_client import MemoryClient
from models import (
    AnthropicMessagesBody,
    ChatAssistantTurn,
    ChatBody,
    ChatMessage,
    ChatTool,
    ChatToolFunction,
    ChatToolResultTurn,
    KeyGenerateBody,
    LiteLLMParamsBody,
    MemoryCaptureBody,
    MemoryEntriesData,
    MemoryEntryParams,
    MemoryLegacyParams,
    MemoryLegacyRows,
    MemoryPolicyBody,
    MemoryResponsesBody,
    MemoryStreamEvent,
    MemoryWireResponse,
    TeamNewBody,
    UserNewBody,
)

pytestmark = pytest.mark.e2e


@dataclass(frozen=True)
class MemorySubjects:
    owner: str
    sibling: str
    outsider: str
    user_id: str
    team_id: str


@dataclass(frozen=True, slots=True)
class MemoryModels:
    chat: str
    messages: str


@pytest.fixture
def memory_models(client: ManagementClient, resources: ResourceManager) -> MemoryModels:
    def register(name: str, model: str, credential: str) -> str:
        alias: Final = f"e2e-memory-{name}-{unique_marker()}"
        identifier: Final = client.proxy.create_model(
            alias,
            LiteLLMParamsBody(
                model=model,
                api_key=credential,
                api_base=os.environ.get("E2E_MEMORY_API_BASE"),
            ),
        )
        resources.defer(lambda: client.proxy.delete_model(identifier))
        return alias

    return MemoryModels(
        chat=register(
            "chat", os.environ.get("E2E_MEMORY_CHAT_MODEL", "openai/gpt-5.6-sol"), "os.environ/OPENAI_API_KEY"
        ),
        messages=register(
            "messages",
            os.environ.get("E2E_MEMORY_MESSAGES_MODEL", "anthropic/claude-haiku-4-5"),
            "os.environ/ANTHROPIC_API_KEY",
        ),
    )


@pytest.fixture
def memory(client: ManagementClient) -> MemoryClient:
    return MemoryClient(client.proxy)


@pytest.fixture
def subjects(client: ManagementClient, memory: MemoryClient, resources: ResourceManager) -> MemorySubjects:
    marker: Final = unique_marker()
    owner: Final = client.create_user(
        UserNewBody(
            user_email=f"memory-owner-{marker}@example.invalid", user_role="internal_user", auto_create_key=False
        )
    )
    resources.defer(lambda: client.delete_user_strict(owner))
    other: Final = client.create_user(
        UserNewBody(
            user_email=f"memory-other-{marker}@example.invalid", user_role="internal_user", auto_create_key=False
        )
    )
    resources.defer(lambda: client.delete_user_strict(other))
    team: Final = client.create_team(TeamNewBody(team_alias=f"memory-{marker}"))
    resources.defer(lambda: client.delete_team(team))
    client.add_team_member(team, owner)
    client.add_team_member(team, other)

    def create_key(user: str) -> str:
        key: Final = unwrap(
            client.generate_key(KeyGenerateBody(user_id=user, team_id=team, models=[], max_parallel_requests=1))
        ).key
        resources.defer(lambda: client.delete_key_strict(key))
        return key

    keys: Final = tuple(create_key(user) for user in (owner, owner, other))
    policy: Final = unwrap(
        memory.set_policy(MemoryPolicyBody(target_type="team", target_id=team, activation="automatic"))
    )
    resources.defer(lambda: memory.delete_policy(policy.policy_id))
    resources.defer(lambda: memory.cleanup_user_entries(owner))
    resources.defer(lambda: memory.cleanup_user_entries(other))
    resources.defer(lambda: memory.preference(keys[0], False))
    resources.defer(lambda: memory.preference(keys[2], False))
    return MemorySubjects(owner=keys[0], sibling=keys[1], outsider=keys[2], user_id=owner, team_id=team)


def _fact(marker: str) -> MemoryCaptureBody:
    return MemoryCaptureBody(
        key=f"release-{marker}",
        title="Demo project codename",
        content=f"The demo project codename is {marker}",
        evidence="Synthetic fact supplied by the authenticated test user",
    )


def _assert_denied(result: object) -> None:
    assert not isinstance(result, Success), "An unauthorized memory operation succeeded"
    assert "403" in str(result) or "unauthorized" in str(result).lower() or "404" in str(result), result


class TestMemoryV2:
    @pytest.mark.covers("mgmt.memory_v2.gateway.capture_recall")
    @pytest.mark.parametrize("endpoint", ["chat", "responses", "messages"])
    @pytest.mark.parametrize("stream", [False, True])
    def test_gateway_stores_and_recalls_without_client_memory_tools(
        self,
        client: ManagementClient,
        memory: MemoryClient,
        subjects: MemorySubjects,
        memory_models: MemoryModels,
        endpoint: str,
        stream: bool,
    ) -> None:
        marker: Final = f"copper-{unique_marker()}"
        seed: Final = unwrap(
            client.proxy.chat(
                subjects.owner,
                ChatBody(
                    model=memory_models.chat,
                    max_tokens=1200,
                    messages=[
                        ChatMessage(
                            role="user",
                            content=f"Remember this durable preference for future conversations: my demo project codename is {marker}. Confirm briefly.",
                        )
                    ],
                ),
            )
        )
        assert seed.choices
        stored: Final = memory.entries(subjects.owner)
        assert any(marker in entry.content for entry in stored), stored
        prompt: Final = "What is my demo project codename? Return the exact word only."
        model: Final = memory_models.messages if endpoint == "messages" else memory_models.chat
        body: Final = (
            AnthropicMessagesBody(
                model=model, messages=[ChatMessage(role="user", content=prompt)], max_tokens=1200, stream=stream
            )
            if endpoint == "messages"
            else MemoryResponsesBody(model=model, input=prompt, stream=stream)
            if endpoint == "responses"
            else ChatBody(
                model=model, messages=[ChatMessage(role="user", content=prompt)], max_tokens=1200, stream=stream
            )
        )
        path: Final = {"messages": "/v1/messages", "responses": "/v1/responses", "chat": "/v1/chat/completions"}[
            endpoint
        ]
        response: Final = client.proxy.transport.send(
            path, headers=client.proxy.transport.bearer(subjects.owner), json=body, stream=stream
        )
        assert response.status_code == 200, response.body
        assert response.stream_error is None, response.stream_error
        output: Final = (
            "".join(MemoryStreamEvent.model_validate_json(event).text for event in response.stream_events)
            if stream
            else response.body
        )
        assert marker in output, output
        if stream:
            assert response.is_streaming
            assert response.chunks > 1
            events: Final = tuple(MemoryStreamEvent.model_validate_json(event) for event in response.stream_events)
            assert not any(event.has_memory_tools for event in events)
            if endpoint == "responses":
                assert all(event.response.instructions is None for event in events if event.response)
        else:
            public: Final = MemoryWireResponse.model_validate_json(response.body)
            assert not public.has_memory_tools
            if endpoint == "responses":
                assert public.instructions is None
        assert memory.entries(subjects.sibling) == []
        assert memory.entries(subjects.outsider) == []

    @pytest.mark.covers("mgmt.memory_v2.policy.opt_in")
    def test_admin_selects_opt_in_or_automatic_and_key_override_wins(
        self,
        client: ManagementClient,
        memory: MemoryClient,
        subjects: MemorySubjects,
        resources: ResourceManager,
        memory_models: MemoryModels,
    ) -> None:
        unwrap(memory.set_policy(MemoryPolicyBody(target_type="team", target_id=subjects.team_id, activation="opt_in")))
        assert not memory.status(subjects.owner).active
        marker: Final = f"unsaved-{unique_marker()}"
        unwrap(
            client.proxy.chat(
                subjects.owner,
                ChatBody(
                    model=memory_models.chat,
                    max_tokens=100,
                    messages=[
                        ChatMessage(
                            role="user", content=f"Remember that my verification word is {marker}. Confirm briefly."
                        )
                    ],
                ),
            )
        )
        assert memory.entries(subjects.owner) == []
        memory.preference(subjects.owner, True)
        assert memory.status(subjects.owner).active
        assert memory.status(subjects.sibling).active
        assert not memory.status(subjects.outsider).active
        saved: Final = unwrap(memory.capture(subjects.owner, _fact(unique_marker())))
        assert saved.memory_id in [entry.memory_id for entry in memory.entries(subjects.owner)]
        key_policy: Final = memory.policy_for_key(subjects.owner, "disabled")
        resources.defer(lambda: memory.delete_policy(key_policy.policy_id))
        assert not memory.status(subjects.owner).active
        assert memory.status(subjects.sibling).active
        memory.policy_for_key(subjects.owner, "automatic")
        memory.preference(subjects.owner, False)
        assert memory.status(subjects.owner).active
        assert not memory.status(subjects.sibling).active

    @pytest.mark.covers("mgmt.memory_v2.entries.isolation")
    def test_private_entries_are_isolated_even_for_sibling_keys_and_legacy_api(
        self, client: ManagementClient, memory: MemoryClient, subjects: MemorySubjects
    ) -> None:
        marker: Final = unique_marker()
        saved: Final = unwrap(memory.capture(subjects.owner, _fact(marker)))
        assert memory.entries(subjects.owner)[0].memory_id == saved.memory_id
        for key in (subjects.sibling, subjects.outsider):
            assert memory.entries(key) == []
            _assert_denied(memory.delete_entry(key, saved.memory_id))
            result = client.proxy.transport.get(
                "/v2/memory/entries",
                headers=client.proxy.transport.bearer(key),
                params=MemoryEntryParams(key_id=hashlib.sha256(subjects.owner.encode()).hexdigest()),
                response_type=MemoryEntriesData,
            )
            _assert_denied(result)
        legacy: Final = client.proxy.transport.get(
            "/v1/memory",
            headers=client.proxy.transport.bearer(subjects.sibling),
            params=MemoryLegacyParams(),
            response_type=MemoryLegacyRows,
        )
        if isinstance(legacy, Success):
            assert saved.memory_id not in [row.memory_id for row in legacy.data.memories]
        assert memory.entries(subjects.owner)[0].content == saved.content

    @pytest.mark.covers("mgmt.memory_v2.policy.admin_only")
    def test_members_cannot_enable_or_broaden_memory(self, memory: MemoryClient, subjects: MemorySubjects) -> None:
        before: Final = memory.status(subjects.owner)
        for target_type, target_id in (("gateway", "*"), ("team", subjects.team_id), ("user", subjects.user_id)):
            body = MemoryPolicyBody.model_validate(
                {"target_type": target_type, "target_id": target_id, "activation": "automatic", "scope": "team"}
            )
            _assert_denied(memory.set_policy(body, caller=subjects.owner))
        assert memory.status(subjects.owner) == before

    @pytest.mark.covers("mgmt.memory_v2.entries.correction_delete")
    def test_corrections_require_current_revision_and_deleted_memory_is_not_recalled(
        self, client: ManagementClient, memory: MemoryClient, subjects: MemorySubjects, memory_models: MemoryModels
    ) -> None:
        original: Final = _fact(unique_marker())
        saved: Final = unwrap(memory.capture(subjects.owner, original))
        corrected_word: Final = f"corrected-{unique_marker()}"
        correction: Final = MemoryCaptureBody(
            key=original.key,
            title=original.title,
            content=f"The demo project codename is {corrected_word}",
            evidence="The user corrected the earlier word",
            expected_revision=saved.updated_at,
        )
        updated: Final = unwrap(memory.capture(subjects.owner, correction))
        assert updated.memory_id == saved.memory_id
        stale: Final = memory.capture(
            subjects.owner, original.model_copy(update={"expected_revision": saved.updated_at})
        )
        assert not isinstance(stale, Success)
        assert "409" in str(stale), stale
        assert memory.entries(subjects.owner)[0].content == correction.content
        unwrap(memory.delete_entry(subjects.owner, saved.memory_id))
        assert memory.entries(subjects.owner) == []
        response: Final = unwrap(
            client.proxy.chat(
                subjects.owner,
                ChatBody(
                    model=memory_models.chat,
                    max_tokens=200,
                    messages=[
                        ChatMessage(
                            role="user", content="What is my demo project codename? If it is unknown, say unknown."
                        )
                    ],
                ),
            )
        )
        assert corrected_word not in response.model_dump_json()

    @pytest.mark.covers("mgmt.memory_v2.gateway.client_tools")
    def test_client_tool_call_and_continuation_remain_owned_by_client(
        self, client: ManagementClient, memory: MemoryClient, subjects: MemorySubjects, memory_models: MemoryModels
    ) -> None:
        marker: Final = unique_marker()
        unwrap(memory.capture(subjects.owner, _fact(marker)))
        tool: Final = ChatTool(
            function=ChatToolFunction(
                name="verify_release",
                description="Verify a release using the user's verification word",
                parameters={"type": "object", "properties": {"word": {"type": "string"}}, "required": ["word"]},
            )
        )
        prompt: Final = ChatMessage(
            role="user",
            content="Use verify_release with my demo project codename. After the tool returns, save its verification result in memory with the tool as your evidence, then report it.",
        )
        response: Final = unwrap(
            client.proxy.chat(
                subjects.owner,
                ChatBody(
                    model=memory_models.chat, messages=[prompt], tools=[tool], tool_choice="required", max_tokens=1200
                ),
            )
        )
        message: Final = response.choices[0].message
        assert message is not None
        calls: Final = message.tool_calls
        assert calls and len(calls) == 1
        call: Final = calls[0]
        assert call.function.name == "verify_release"
        assert marker in (call.function.arguments or "")
        assert call.id
        result_marker: Final = f"verified-{unique_marker()}"
        followup: Final = unwrap(
            client.proxy.chat(
                subjects.owner,
                ChatBody(
                    model=memory_models.chat,
                    max_tokens=1200,
                    tools=[tool],
                    messages=[
                        prompt,
                        ChatAssistantTurn(
                            content=message.content, reasoning_content=message.reasoning_content, tool_calls=calls
                        ),
                        ChatToolResultTurn(tool_call_id=call.id, content=result_marker),
                    ],
                ),
            )
        )
        assert result_marker in followup.model_dump_json()
        assert any(result_marker in entry.content for entry in memory.entries(subjects.owner))

    @pytest.mark.covers("mgmt.memory_v2.gateway.billing")
    def test_memory_tool_rounds_are_charged_once_to_the_calling_key(
        self, client: ManagementClient, memory: MemoryClient, subjects: MemorySubjects, memory_models: MemoryModels
    ) -> None:
        marker: Final = unique_marker()
        response: Final = unwrap(
            client.proxy.chat(
                subjects.owner,
                ChatBody(
                    model=memory_models.chat,
                    max_tokens=1200,
                    messages=[
                        ChatMessage(
                            role="user", content=f"Remember that my demo project codename is {marker}. Confirm briefly."
                        )
                    ],
                ),
            )
        )
        assert response.choices
        assert any(marker in row.content for row in memory.entries(subjects.owner))
        rows: Final = client.proxy.poll_logs_for_key(subjects.owner, min_rows=2)
        assert 2 <= len(rows) <= 8, rows
        assert len({row.request_id for row in rows}) == len(rows), rows
        assert all(row.api_key == hashlib.sha256(subjects.owner.encode()).hexdigest() for row in rows), rows
        assert all(row.user == subjects.user_id and row.team_id == subjects.team_id for row in rows), rows
        assert all(row.prompt_tokens and row.completion_tokens for row in rows), rows
        assert all(row.spend is not None and row.spend > 0 for row in rows), rows
