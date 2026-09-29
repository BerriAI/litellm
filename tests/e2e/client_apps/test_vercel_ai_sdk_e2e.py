"""The Vercel AI SDK (`ai` + `@ai-sdk/openai`, pinned in this suite's
package.json) runs one `streamText` turn against the proxy on a virtual key,
once over `/v1/responses` and once over `/v1/chat/completions`: the model calls
a local tool whose result is a secret word only that tool knows, then answers
with the word. Client side the summary carries the streamed text deltas, the
tool call, the tool result, and the final text; proxy side the spend rows carry
real spend, `stream: true`, and the request body shape of the API used."""

from __future__ import annotations

import pytest

from client_apps_client import ClientAppsClient, unwrap_run
from client_apps_models import VercelApi
from e2e_config import unique_marker
from spend_rows import recorded_request, streamed_tool_turn_rows

pytestmark = pytest.mark.e2e


class TestVercelAiSdk:
    @pytest.mark.parametrize(
        "api",
        [
            pytest.param(
                "responses", marks=pytest.mark.covers("other.client_apps.vercel_ai_sdk_responses.tool_turn_streams")
            ),
            pytest.param(
                "chat", marks=pytest.mark.covers("other.client_apps.vercel_ai_sdk_chat_completions.tool_turn_streams")
            ),
        ],
    )
    def test_streams_a_tool_turn(
        self, client: ClientAppsClient, scoped_key: str, openai_model: str, api: VercelApi
    ) -> None:
        secret_word = f"pong-{unique_marker()}"
        turn = unwrap_run(
            client.run_vercel_ai_sdk(key=scoped_key, model=openai_model, api=api, secret_word=secret_word)
        )

        assert turn.errors == [], turn
        assert [call.tool_name for call in turn.tool_calls] == ["reveal_secret_word"], turn
        assert [result.output.word for result in turn.tool_results] == [secret_word], turn
        assert secret_word in turn.text, turn
        assert turn.text_deltas >= 1, turn
        assert turn.steps == 2, turn
        assert turn.finish_reason == "stop", turn

        rows = streamed_tool_turn_rows(client.proxy, scoped_key)
        bodies = [recorded_request(row) for row in rows]
        match api:
            case "responses":
                assert all(body.input is not None for body in bodies), rows
            case "chat":
                assert all(body.messages is not None for body in bodies), rows
