"""Client for the guardrails e2e suite: register global (default-on) guardrails
and chat through them on the shared ProxyClient so resources.defer cleans up.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from e2e_config import POLL_INTERVAL, POLL_TIMEOUT, unique_marker
from e2e_http import NoBody, Result, Success, unwrap
from lifecycle import ResourceManager
from models import (
    ChatBody,
    ChatMessage,
    ChatResponse,
    KeyGenerateBody,
    LiteLLMParamsBody,
    TeamDeleteBody,
    TeamInfoParams,
    TeamInfoResponse,
    TeamMetadata,
    TeamNewBody,
    TeamNewResponse,
)
from proxy_client import ProxyClient

GuardrailMode = Literal["pre_call", "post_call", "during_call", "logging_only"]
BlockedWordAction = Literal["BLOCK", "MASK"]


class BlockedWordBody(BaseModel):
    keyword: str
    action: BlockedWordAction


class GuardrailParamsBase(BaseModel):
    mode: GuardrailMode
    default_on: bool


class ContentFilterParamsBody(GuardrailParamsBase):
    guardrail: Literal["litellm_content_filter"] = "litellm_content_filter"
    blocked_words: list[BlockedWordBody]


class BedrockGuardrailParamsBody(GuardrailParamsBase):
    guardrail: Literal["bedrock"] = "bedrock"
    guardrailIdentifier: str
    guardrailVersion: str
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region_name: str | None = None


class OpenAIModerationParamsBody(GuardrailParamsBase):
    guardrail: Literal["openai_moderation"] = "openai_moderation"
    api_key: str | None = None
    model: str | None = None


class PresidioParamsBody(GuardrailParamsBase):
    guardrail: Literal["presidio"] = "presidio"
    presidio_analyzer_api_base: str | None = None
    presidio_anonymizer_api_base: str | None = None
    # apply_to_output masks PII the model itself emitted, which also makes the
    # guardrail run post_call.
    apply_to_output: bool | None = None


class BlockCodeExecutionParamsBody(GuardrailParamsBase):
    guardrail: Literal["block_code_execution"] = "block_code_execution"


GuardrailParamsBody = (
    ContentFilterParamsBody
    | BedrockGuardrailParamsBody
    | OpenAIModerationParamsBody
    | PresidioParamsBody
    | BlockCodeExecutionParamsBody
)


class GuardrailSpecBody(BaseModel):
    guardrail_name: str
    litellm_params: GuardrailParamsBody


class GuardrailCreateBody(BaseModel):
    guardrail: GuardrailSpecBody


class GuardrailCreateResponse(BaseModel):
    guardrail_id: str


class ApplyGuardrailRequest(BaseModel):
    guardrail_name: str
    text: str
    language: str | None = None
    input_type: str = "request"


class ApplyGuardrailResponse(BaseModel):
    response_text: str


@dataclass(frozen=True, slots=True)
class GuardrailsClient:
    proxy: ProxyClient

    def create_content_filter_guardrail(self, name: str, blocked_keyword: str) -> str:
        return unwrap(
            self.proxy.transport.post(
                "/guardrails",
                headers=self.proxy.transport.master,
                json=GuardrailCreateBody(
                    guardrail=GuardrailSpecBody(
                        guardrail_name=name,
                        litellm_params=ContentFilterParamsBody(
                            mode="pre_call",
                            default_on=True,
                            blocked_words=[
                                BlockedWordBody(keyword=blocked_keyword, action="BLOCK")
                            ],
                        ),
                    )
                ),
                response_type=GuardrailCreateResponse,
            )
        ).guardrail_id

    def create_bedrock_guardrail(
        self,
        name: str,
        *,
        identifier: str,
        version: str,
    ) -> str:
        return unwrap(
            self.proxy.transport.post(
                "/guardrails",
                headers=self.proxy.transport.master,
                json=GuardrailCreateBody(
                    guardrail=GuardrailSpecBody(
                        guardrail_name=name,
                        litellm_params=BedrockGuardrailParamsBody(
                            mode="pre_call",
                            default_on=True,
                            guardrailIdentifier=identifier,
                            guardrailVersion=version,
                            aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
                            aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
                            aws_region_name="os.environ/AWS_REGION",
                        ),
                    )
                ),
                response_type=GuardrailCreateResponse,
            )
        ).guardrail_id

    def create_backend_model(self, resources: ResourceManager, prefix: str = "e2e-guard-backend") -> str:
        """Register a gemini chat deployment for a guardrail test to run against
        (deleted on teardown). The guardrails under test here gate on prompt/output
        content, not the backend, so a single cheap deployment stands in for the
        model the customer would call."""
        model_name = f"{prefix}-{unique_marker()}"
        model_id = self.proxy.create_model(
            model_name,
            LiteLLMParamsBody(model="gemini/gemini-2.5-flash", api_key="os.environ/GEMINI_API_KEY"),
        )
        resources.defer(lambda: self.proxy.delete_model(model_id))
        return model_name

    def register(self, name: str, params: GuardrailParamsBody) -> str:
        """Register any guardrail via POST /guardrails and return its id. New
        built-ins register with default_on=False and are opted into per request
        via the chat body's `guardrails` list, so one guardrail under test never
        intercepts unrelated traffic on the shared proxy."""
        return unwrap(
            self.proxy.transport.post(
                "/guardrails",
                headers=self.proxy.transport.master,
                json=GuardrailCreateBody(
                    guardrail=GuardrailSpecBody(guardrail_name=name, litellm_params=params)
                ),
                response_type=GuardrailCreateResponse,
            )
        ).guardrail_id

    def delete_guardrail(self, guardrail_id: str) -> None:
        _ = self.proxy.transport.delete(
            f"/guardrails/{guardrail_id}",
            headers=self.proxy.transport.master,
            json=NoBody(),
            response_type=NoBody,
        )

    def create_team_opted_out_of_global_guardrails(self, alias: str) -> str:
        team_id = unwrap(
            self.proxy.transport.post(
                "/team/new",
                headers=self.proxy.transport.master,
                json=TeamNewBody(
                    team_alias=alias,
                    metadata=TeamMetadata(disable_global_guardrails=True),
                ),
                response_type=TeamNewResponse,
            )
        ).team_id
        self._await_team(team_id)
        return team_id

    def delete_team(self, team_id: str) -> None:
        _ = self.proxy.transport.post(
            "/team/delete",
            headers=self.proxy.transport.master,
            json=TeamDeleteBody(team_ids=[team_id]),
            response_type=NoBody,
        )

    def create_key_in_team(self, team_id: str) -> str:
        return self.proxy.generate_key(
            KeyGenerateBody(team_id=team_id, user_id="e2e-guardrails-user")
        )

    def chat(
        self,
        key: str,
        model: str,
        text: str,
        *,
        guardrails: list[str] | None = None,
        max_tokens: int = 16,
    ) -> Result[ChatResponse]:
        """Drive a chat call, optionally opting into named guardrails for this
        request only (the per-request `guardrails` selector). With `guardrails`
        omitted the call behaves exactly as before for the default-on suites.
        `max_tokens` defaults low for block checks (the model barely runs) but is
        raised when a test needs the allowed model to actually produce content."""
        return self.proxy.chat(
            key,
            ChatBody(
                model=model,
                messages=[ChatMessage(role="user", content=text)],
                max_tokens=max_tokens,
                guardrails=guardrails,
            ),
        )

    def apply_guardrail(self, key: str, *, name: str, text: str) -> Result[ApplyGuardrailResponse]:
        return self.proxy.transport.post(
            "/guardrails/apply_guardrail",
            headers=self.proxy.transport.bearer(key),
            json=ApplyGuardrailRequest(guardrail_name=name, text=text),
            response_type=ApplyGuardrailResponse,
        )

    def _await_team(self, team_id: str) -> None:
        deadline = time.monotonic() + POLL_TIMEOUT
        last: Result[TeamInfoResponse] | None = None
        while time.monotonic() < deadline:
            last = self.proxy.transport.get(
                "/team/info",
                headers=self.proxy.transport.master,
                params=TeamInfoParams(team_id=team_id),
                response_type=TeamInfoResponse,
            )
            if isinstance(last, Success):
                return
            time.sleep(POLL_INTERVAL)
        raise AssertionError(
            f"team {team_id!r} was created but /team/info never returned it: {last}"
        )


def build_client(proxy: ProxyClient) -> GuardrailsClient:
    return GuardrailsClient(proxy=proxy)
