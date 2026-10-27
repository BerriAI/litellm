import asyncio
import base64
import datetime
import json
import os
import unittest
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, List, Optional, Tuple, cast
from unittest.mock import ANY, MagicMock, Mock, patch

import httpx
import pytest
import respx

import litellm
from pydantic import TypeAdapter
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.types.utils import StandardLoggingPayload


@dataclass(frozen=True, slots=True)
class _CachedVertexCredentials:
    token: str
    quota_project_id: str | None
    expired: bool = False

    def refresh(self, request: object) -> None:
        return None


@pytest.fixture(autouse=True)
def _httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.mark.asyncio
async def test_construct_request_headers_project_id_from_env(monkeypatch):
    """Test that construct_request_headers uses GCS_PUBSUB_PROJECT_ID environment variable."""
    from litellm.integrations.gcs_pubsub.pub_sub import GcsPubSubLogger

    # Set up test environment variable
    test_project_id = "test-project-123"
    monkeypatch.setenv("GCS_PUBSUB_PROJECT_ID", test_project_id)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.premium_user",
        True,
    )

    try:
        # Create handler with no project_id
        handler = GcsPubSubLogger(
            topic_id="test-topic", credentials_path="test-path.json"
        )

        # Mock the Vertex AI auth calls
        mock_auth_header = "mock-auth-header"
        mock_token = "mock-token"

        with patch(
            "litellm.vertex_chat_completion.ensure_access_token_async"
        ) as mock_ensure_token:
            mock_ensure_token.return_value = (mock_auth_header, test_project_id)

            with patch(
                "litellm.vertex_chat_completion._get_token_and_url"
            ) as mock_get_token:
                mock_get_token.return_value = (mock_token, "mock-url")

                # Call construct_request_headers
                headers = await handler.construct_request_headers()

                # Verify headers
                assert headers == {
                    "Authorization": f"Bearer {mock_token}",
                    "Content-Type": "application/json",
                }

                # Verify ensure_access_token_async was called with correct project_id
                mock_ensure_token.assert_called_once_with(
                    credentials="test-path.json",
                    project_id=test_project_id,
                    custom_llm_provider="vertex_ai",
                )
    finally:
        # Clean up environment variable
        del os.environ["GCS_PUBSUB_PROJECT_ID"]


def _authorized_user_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, project_id: str) -> str:
    credentials_path: Final = tmp_path / "google-auth.json"
    credentials_path.write_text(
        json.dumps(
            {
                "type": "authorized_user",
                "client_id": "unit-client",
                "client_secret": "unit-secret",
                "refresh_token": "unit-refresh-token",
                "token": "unit-access-token",
                "token_uri": "https://oauth2.googleapis.com/token",
                "quota_project_id": "unit-project",
                "expiry": "2099-01-01T00:00:00Z",
            }
        )
    )
    credentials_path_string: Final = str(credentials_path)
    credentials: Final = _CachedVertexCredentials(token="unit-access-token", quota_project_id=project_id)
    vertex_chat_completion: Final = litellm.vertex_chat_completion
    monkeypatch.setitem(
        vertex_chat_completion._credentials_project_mapping,
        (credentials_path_string, project_id),
        (credentials, project_id),
    )
    return credentials_path_string


@pytest.mark.asyncio
async def test_publish_message_encodes_the_standard_payload(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    from litellm.integrations.gcs_pubsub.pub_sub import GcsPubSubLogger

    project_id: Final = "unit-project"
    topic_id: Final = "unit-topic"
    url: Final = f"https://pubsub.googleapis.com/v1/projects/{project_id}/topics/{topic_id}:publish"
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
    route: Final = respx_mock.post(url).mock(return_value=httpx.Response(202, json={"messageIds": ["publish-1"]}))
    logger: Final = GcsPubSubLogger(
        project_id=project_id,
        topic_id=topic_id,
        credentials_path=_authorized_user_credentials(monkeypatch, tmp_path, project_id),
    )
    payload: Final = cast(
        StandardLoggingPayload,
        {
            "request_id": "request-1",
            "model": "vertex_ai/gemini-2.5-flash",
            "messages": [{"role": "user", "content": "publish this"}],
            "response": {"choices": [{"message": {"content": "published"}}]},
        },
    )

    response: Final = TypeAdapter(dict[str, list[str]]).validate_python(await logger.publish_message(payload))
    request_json: Final = TypeAdapter(dict[str, list[dict[str, str]]]).validate_json(route.calls.last.request.content)
    decoded_message: Final = TypeAdapter(dict[str, object]).validate_json(
        base64.b64decode(request_json["messages"][0]["data"])
    )

    assert route.call_count == 1
    assert route.calls.last.request.url == url
    assert route.calls.last.request.headers["Authorization"] == "Bearer unit-access-token"
    assert request_json == {
        "messages": [
            {
                "data": base64.b64encode(json.dumps(payload, default=str).encode("utf-8")).decode("utf-8"),
            }
        ]
    }
    assert decoded_message == payload
    assert response == {"messageIds": ["publish-1"]}


def _without_volatile_gcs_payload_fields(value: object, path: str = "") -> object:
    ignored_paths: Final = frozenset(
        {
            "request_id",
            "litellm_call_id",
            "metadata.litellm_call_id",
            "session_id",
            "startTime",
            "endTime",
            "completionStartTime",
            "request_duration_ms",
            "metadata.model_map_information",
            "metadata.usage_object",
            "metadata.cold_storage_object_key",
            "metadata.litellm_overhead_time_ms",
            "metadata.cost_breakdown",
            "metadata.autorouter_savings",
            "metadata.eval_information",
        }
    )
    if isinstance(value, Mapping):
        mapping: Final = TypeAdapter(dict[str, object]).validate_python(value)
        return MappingProxyType(
            {
                key: _without_volatile_gcs_payload_fields(item, f"{path}.{key}" if path else key)
                for key, item in mapping.items()
                if (f"{path}.{key}" if path else key) not in ignored_paths
            }
        )
    if isinstance(value, list):
        items: Final = TypeAdapter(list[object]).validate_python(value)
        return tuple(
            _without_volatile_gcs_payload_fields(item, f"{path}[{index}]") for index, item in enumerate(items)
        )
    return value


def _with_decoded_metadata(payload: Mapping[str, object]) -> dict[str, object]:
    metadata: Final = payload["metadata"]
    return {**payload, "metadata": json.loads(metadata) if isinstance(metadata, str) else metadata}


@pytest.mark.asyncio
async def test_v1_success_event_publishes_the_spend_log_fixture(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter, tmp_path: Path
) -> None:
    from litellm.integrations.gcs_pubsub.pub_sub import GcsPubSubLogger

    project_id: Final = "unit-project"
    topic_id: Final = "unit-topic"
    url: Final = f"https://pubsub.googleapis.com/v1/projects/{project_id}/topics/{topic_id}:publish"
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
    monkeypatch.setattr(litellm, "gcs_pub_sub_use_v1", True)
    route: Final = respx_mock.post(url).mock(return_value=httpx.Response(202, json={"messageIds": ["publish-2"]}))
    logger: Final = GcsPubSubLogger(
        project_id=project_id,
        topic_id=topic_id,
        credentials_path=_authorized_user_credentials(monkeypatch, tmp_path, project_id),
    )
    monkeypatch.setattr(litellm, "callbacks", [logger])
    await litellm.acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello, world!"}],
        mock_response="hi",
    )
    await asyncio.sleep(0)
    await GLOBAL_LOGGING_WORKER.flush()
    await logger.async_send_batch()

    request_json: Final = TypeAdapter(dict[str, list[dict[str, str]]]).validate_json(route.calls.last.request.content)
    decoded_payload: Final = TypeAdapter(dict[str, object]).validate_json(
        base64.b64decode(request_json["messages"][0]["data"])
    )
    expected_payload: Final = TypeAdapter(dict[str, object]).validate_json(
        Path(__file__).with_name("spend_logs_payload.json").read_text()
    )
    assert route.call_count == 1
    assert route.calls.last.request.url == url
    assert _without_volatile_gcs_payload_fields(_with_decoded_metadata(decoded_payload)) == (
        _without_volatile_gcs_payload_fields(_with_decoded_metadata(expected_payload))
    )
