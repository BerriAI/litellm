"""Live e2e pins for the complexity router's context-window escalation when a tier's
window is declared only through ``model_info.base_model``.

A Bedrock application inference profile ARN has no cost-map row of its own, so the
SIMPLE tier's window exists only through ``base_model``. ``Router.get_router_model_info``
read base_model for azure deployments alone and resolved every other provider's
deployment by its own ``model:``, so the ARN came back with no window, the escalation
gate treated an unknown window as able to hold anything, and an oversized request the
classifier scored SIMPLE stayed on the 200k model and 400'd at Bedrock with "prompt is
too long" instead of moving to the 1M MEDIUM tier.

The explicit-window control declares ``max_input_tokens`` below its model's own row and
sends a request that fits the row but not the declaration, so it escalates only because
a declared window is honored. It lives on its own backend model: the router registers a
deployment's ``model_info`` under the shared cost-map key of its ``model:``, so a declared
window on the ARN would leak to the base_model deployment and hide the bug. A short
request on the base_model router proves the gate only moves what does not fit.

Every deployment is registered via /model/new (stage has no static config for these)
and the served deployment is read back from the spend log's ``model``, which stores
either the registered alias or the provider-prefixed form. The prompts are plain prose
with a "how many" ask, which keeps the heuristic classifier on SIMPLE, and the router
estimates their size as chars/4.
"""

import os
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Final

import pytest

from e2e_config import unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from models import (
    ChatBody,
    ChatMessage,
    ChatResponse,
    KeyGenerateBody,
    LiteLLMParamsBody,
    ModelInfoBody,
    ModelNewBody,
    SpendLogRow,
)
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e

SIMPLE_BACKEND: Final = os.environ.get(
    "E2E_BEDROCK_INFERENCE_PROFILE_ARN",
    "bedrock/converse/arn:aws:bedrock:us-west-2:888602223428:application-inference-profile/sccejsk6r44v",
)
SIMPLE_BASE_MODEL: Final = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
EXPLICIT_WINDOW_BACKEND: Final = "bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0"
DECLARED_WINDOW_TOKENS: Final = 60_000
UPPER_BACKEND: Final = "bedrock/converse/global.amazon.nova-2-lite-v1:0"
MAX_TOKENS: Final = 8
OVERSIZED_PROMPT_CHARS: Final = 1_000_000
BEYOND_DECLARED_WINDOW_PROMPT_CHARS: Final = 400_000
FILLER: Final = (
    "On the morning of day {n}, the garden club met beside the old mill and talked about which flowers "
    "to grow along the river walk. Everyone brought warm bread and shared stories about the cold winter "
    "and the first robin of spring.\n"
)
ASK: Final = "How many days are written above? Reply with the single word OK."


@dataclass(frozen=True, slots=True)
class EscalationRouters:
    base_model_router: str
    explicit_window_router: str
    simple_from_base_model: str
    simple_with_explicit_window: str
    upper: str


def _bedrock_params(model: str) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=model,
        aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
        aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
        aws_region_name="us-west-2",
    )


def _router_params(simple: str, upper: str) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model="auto_router/complexity_router",
        complexity_router_config={
            "classifier_type": "heuristic",
            "enable_context_window_escalation": True,
            "tiers": {"SIMPLE": simple, "MEDIUM": upper, "COMPLEX": upper, "REASONING": upper},
        },
    )


def _prompt_of(chars: int) -> str:
    marker: Final = unique_marker()
    line_count: Final = -(-chars // len(FILLER.format(n=f"1 ({marker})")))
    lines: Final = tuple(FILLER.format(n=f"{n} ({marker})") for n in range(1, line_count + 1))
    return "".join(lines) + "\n" + ASK


def _chat_body(model: str, content: str) -> ChatBody:
    return ChatBody(model=model, messages=[ChatMessage(role="user", content=content)], max_tokens=MAX_TOKENS)


def _key_for(proxy: ProxyClient, resources: ResourceManager, models: list[str]) -> str:
    key: Final = proxy.generate_key(KeyGenerateBody(models=models, user_id="e2e-context-window-base-model"))
    resources.defer(lambda: proxy.delete_key(key))
    return key


def _served_by(rows: list[SpendLogRow], alias: str, backend: str, context: str) -> None:
    served: Final = tuple(row.model for row in rows)
    assert served and all(model in {alias, backend} for model in served), (
        f"{context}: expected every request to be served by {alias} ({backend}), spend logs show {served}"
    )


def _answered(response: ChatResponse) -> None:
    message: Final = response.choices[0].message if response.choices else None
    assert message is not None and message.content, f"expected an answer, got {response}"


@pytest.fixture(scope="class")
def router_stack() -> Iterator[ExitStack]:
    with ExitStack() as stack:
        yield stack


@pytest.fixture(scope="class")
def routers(proxy: ProxyClient, router_stack: ExitStack) -> EscalationRouters:
    marker: Final = unique_marker()
    named: Final = EscalationRouters(
        base_model_router=f"e2e-cw-router-base-{marker}",
        explicit_window_router=f"e2e-cw-router-explicit-{marker}",
        simple_from_base_model=f"e2e-cw-simple-base-{marker}",
        simple_with_explicit_window=f"e2e-cw-simple-explicit-{marker}",
        upper=f"e2e-cw-upper-{marker}",
    )
    registrations: Final = (
        ModelNewBody(
            model_name=named.simple_from_base_model,
            litellm_params=_bedrock_params(SIMPLE_BACKEND),
            model_info=ModelInfoBody(base_model=SIMPLE_BASE_MODEL),
        ),
        ModelNewBody(
            model_name=named.simple_with_explicit_window,
            litellm_params=_bedrock_params(EXPLICIT_WINDOW_BACKEND),
            model_info=ModelInfoBody(max_input_tokens=DECLARED_WINDOW_TOKENS),
        ),
        ModelNewBody(model_name=named.upper, litellm_params=_bedrock_params(UPPER_BACKEND), model_info=ModelInfoBody()),
        ModelNewBody(
            model_name=named.base_model_router,
            litellm_params=_router_params(named.simple_from_base_model, named.upper),
            model_info=ModelInfoBody(),
        ),
        ModelNewBody(
            model_name=named.explicit_window_router,
            litellm_params=_router_params(named.simple_with_explicit_window, named.upper),
            model_info=ModelInfoBody(),
        ),
    )
    for body in registrations:
        router_stack.callback(proxy.delete_model, proxy.register_model(body))
    return named


class TestContextWindowEscalationFromBaseModel:
    @pytest.mark.covers("reliability.routing.context_window_escalation.escalates_from_base_model_window")
    def test_base_model_window_escalates(
        self, proxy: ProxyClient, resources: ResourceManager, routers: EscalationRouters
    ) -> None:
        key: Final = _key_for(
            proxy, resources, [routers.base_model_router, routers.simple_from_base_model, routers.upper]
        )

        response: Final = unwrap(
            proxy.chat(key, _chat_body(routers.base_model_router, _prompt_of(OVERSIZED_PROMPT_CHARS)))
        )

        _answered(response)
        _served_by(proxy.poll_logs_for_key(key), routers.upper, UPPER_BACKEND, "oversized ask, window from base_model")

    @pytest.mark.covers("reliability.routing.context_window_escalation.escalates_from_explicit_window")
    def test_explicit_window_escalates(
        self, proxy: ProxyClient, resources: ResourceManager, routers: EscalationRouters
    ) -> None:
        key: Final = _key_for(
            proxy, resources, [routers.explicit_window_router, routers.simple_with_explicit_window, routers.upper]
        )

        response: Final = unwrap(
            proxy.chat(key, _chat_body(routers.explicit_window_router, _prompt_of(BEYOND_DECLARED_WINDOW_PROMPT_CHARS)))
        )

        _answered(response)
        _served_by(proxy.poll_logs_for_key(key), routers.upper, UPPER_BACKEND, "ask beyond the declared window")

    @pytest.mark.covers("reliability.routing.context_window_escalation.fitting_prompt_stays_on_tier")
    def test_fitting_prompt_stays_on_the_simple_tier(
        self, proxy: ProxyClient, resources: ResourceManager, routers: EscalationRouters
    ) -> None:
        key: Final = _key_for(
            proxy, resources, [routers.base_model_router, routers.simple_from_base_model, routers.upper]
        )

        response: Final = unwrap(proxy.chat(key, _chat_body(routers.base_model_router, f"say hello {unique_marker()}")))

        _answered(response)
        _served_by(
            proxy.poll_logs_for_key(key),
            routers.simple_from_base_model,
            SIMPLE_BACKEND,
            "short ask, window from base_model",
        )
