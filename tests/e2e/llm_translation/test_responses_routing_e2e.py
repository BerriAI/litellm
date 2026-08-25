"""Live e2e: deployment-level mode=responses drives routing and health checks.

Two regressions around the Responses API bridge, both hit through models
registered via /model/new rather than shipped in the static cost map:

- routing: /chat/completions on a responses-only deployment used to work by
  model-name matching, then 1.92 moved the decision to a cost-map lookup and
  every model whose entry had not shipped yet started failing with "does not
  support the /v1/chat/completions API". The router now injects the deployment's
  model_info (including mode) into the cost map at registration, so a deployment
  declaring mode=responses must bridge regardless of what the static map says.
  The spend log's call_type distinguishes the two paths: "responses" only when
  the bridge translated the call
- health: POST /health/test_connection with mode=responses used to crash with
  "functools.partial got multiple values for 'acompletion'" (the bridge handed
  litellm.acompletion a second acompletion kwarg). The UI's Test Connection
  button on any responses-mode model hit this as a 500
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, LiteLLMParamsBody
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e

BACKEND_MODEL = "gemini/gemini-2.5-flash"
GEMINI_API_KEY = "os.environ/GEMINI_API_KEY"


class HealthCheckParams(BaseModel):
    model: str


class HealthCheckModelInfo(BaseModel):
    id: str


class HealthCheckBody(BaseModel):
    litellm_params: HealthCheckParams
    model_info: HealthCheckModelInfo
    mode: str


class HealthCheckResult(BaseModel):
    error: str | None = None


class HealthCheckResponse(BaseModel):
    status: str
    result: HealthCheckResult = HealthCheckResult()


def _register_responses_mode_model(proxy: ProxyClient, resources: ResourceManager) -> tuple[str, str]:
    """Register a gemini deployment declaring mode=responses in its model_info,
    the shape a responses-only model (bedrock_mantle gpt-5.x, openai codex)
    takes when added through the UI. Returns (model_name, model_id)."""
    model_name = f"e2e-responses-mode-{unique_marker()}"
    model_id = proxy.create_model(model_name, LiteLLMParamsBody(model=BACKEND_MODEL, api_key=GEMINI_API_KEY))
    resources.defer(lambda: proxy.delete_model(model_id))
    return model_name, model_id


class TestResponsesModeRouting:
    @pytest.mark.covers(
        "llm.chat_completions.gemini.basic.nonstream.bridges_to_responses",
        exercised_on=["chat_completions"],
    )
    def test_chat_completion_bridges_when_deployment_declares_responses_mode(
        self, proxy: ProxyClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        model_name = f"e2e-responses-mode-{unique_marker()}"
        model_id = proxy.create_model(
            model_name,
            LiteLLMParamsBody(model=BACKEND_MODEL, api_key=GEMINI_API_KEY),
            mode="responses",
        )
        resources.defer(lambda: proxy.delete_model(model_id))

        chat = unwrap(
            proxy.chat(
                scoped_key,
                ChatBody(
                    model=model_name,
                    messages=[ChatMessage(role="user", content=f"Reply with only the word ok. {unique_marker()}")],
                    max_tokens=256,
                ),
            )
        )
        assert chat.id and chat.id.startswith("chatcmpl-"), (
            f"the bridged call must still answer in chat-completion shape: id={chat.id!r}"
        )
        assert chat.choices and chat.choices[0].message is not None, (
            f"bridged call returned no choices: {chat}"
        )

        rows = proxy.poll_logs_for_key(scoped_key)
        call_types = {row.call_type for row in rows}
        assert "responses" in call_types, (
            f"a deployment with model_info.mode=responses must be served through the "
            f"Responses bridge (spend log call_type='responses'); the call went down the "
            f"chat path instead: {call_types}"
        )

    @pytest.mark.covers("other.health.test_connection.responses_mode_succeeds")
    def test_health_test_connection_in_responses_mode_succeeds(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        _, model_id = _register_responses_mode_model(proxy, resources)

        # The UI's Test Connection button sends the provider model plus the
        # deployment id; mode=responses selects the aresponses health handler.
        health = unwrap(
            proxy.transport.post(
                "/health/test_connection",
                headers=proxy.transport.master,
                json=HealthCheckBody(
                    litellm_params=HealthCheckParams(model=BACKEND_MODEL),
                    model_info=HealthCheckModelInfo(id=model_id),
                    mode="responses",
                ),
                response_type=HealthCheckResponse,
            )
        )
        error = health.result.error or ""
        assert "acompletion" not in error and "functools.partial" not in error, (
            f"health check crashed on the duplicate-acompletion TypeError the responses "
            f"bridge used to raise: {error}"
        )
        assert health.status == "success", f"responses-mode health check failed: {error}"
