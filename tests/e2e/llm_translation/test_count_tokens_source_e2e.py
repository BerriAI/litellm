from __future__ import annotations

import base64
import io
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

import pytest
from anthropic import Anthropic, BadRequestError, Omit, omit
from anthropic.types import DocumentBlockParam, MessageParam, TextBlockParam, ToolParam
from e2e_config import unique_marker
from e2e_http import unwrap
from e2e_metadata import Capability, Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import (
    ChatMessage,
    CountTokensResponse,
    LiteLLMParamsBody,
    ModelInfoBody,
    ModelNewBody,
    ResponsesInputTokensResponse,
    TokenCounterBody,
)
from proxy_client import ProxyClient
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen.canvas import Canvas
from sdk_clients import NO_PROXY_CACHE, SdkClients

pytestmark = pytest.mark.e2e

BEDROCK_SONNET: Final = "bedrock/us.anthropic.claude-sonnet-5-5"
COST_MAP_FALLBACK: Final = "us.anthropic.claude-opus-5-5"
CONFIGURED_FALLBACK: Final = "us.anthropic.claude-sonnet-5"
ANTHROPIC_SONNET: Final = "anthropic/claude-sonnet-5-5"
BEDROCK_KIMI: Final = "bedrock/us.moonshotai.kimi-k3"
BEDROCK_NOVA: Final = "bedrock/us.amazon.nova-pro-v1:0"
OPENAI_GPT: Final = "openai/gpt-5.6"

COUNT_TOLERANCE: Final = 0.05
MAX_TOKENS: Final = 16
PDF_LINES_PER_PAGE: Final = 45

LARGE_SYSTEM: Final = " ".join(
    f"Rule {n}: answer in plain English, cite clause {n} when it applies, and never invent a figure."
    for n in range(1, 301)
)
LARGE_TOOL: Final[ToolParam] = {
    "name": "search_orders",
    "description": " ".join(
        f"Filter {n} narrows the order search by the attribute recorded in field_{n}." for n in range(1, 151)
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            f"field_{n}": {"type": "string", "description": f"Order attribute number {n}, matched exactly."}
            for n in range(1, 61)
        },
        "required": ["field_1"],
    },
}


@dataclass(frozen=True, slots=True)
class Served:
    model: str
    key: str


@dataclass(frozen=True, slots=True)
class PromptShape:
    name: str
    system: str | None = None
    tool: ToolParam | None = None
    pdf_pages: int = 0


SHAPES: Final = (
    PromptShape(name="baseline"),
    PromptShape(name="large_system", system=LARGE_SYSTEM),
    PromptShape(name="large_tool", tool=LARGE_TOOL),
    PromptShape(name="system_and_tool", system=LARGE_SYSTEM, tool=LARGE_TOOL),
    PromptShape(name="pdf_1_page", pdf_pages=1),
    PromptShape(name="pdf_12_pages", pdf_pages=12),
)
SHAPE_IDS: Final = tuple(shape.name for shape in SHAPES)


def _bedrock(model: str) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=model,
        aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
        aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
        aws_region_name="os.environ/AWS_REGION",
    )


def _serve(proxy: ProxyClient, backend: LiteLLMParamsBody, model_info: ModelInfoBody | None = None) -> Iterator[Served]:
    resources: Final = ResourceManager(client=proxy)
    try:
        model: Final = f"e2e-count-source-{unique_marker()}"
        model_id: Final = proxy.register_model(
            ModelNewBody(model_name=model, litellm_params=backend, model_info=model_info or ModelInfoBody())
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        yield Served(model=model, key=resources.key())
    finally:
        resources.teardown()


@pytest.fixture(scope="module")
def bedrock_sonnet(proxy: ProxyClient) -> Iterator[Served]:
    yield from _serve(proxy, _bedrock(BEDROCK_SONNET))


@pytest.fixture(scope="module")
def bedrock_sonnet_with_configured_fallback(proxy: ProxyClient) -> Iterator[Served]:
    yield from _serve(proxy, _bedrock(BEDROCK_SONNET), ModelInfoBody(token_count_fallback_model=CONFIGURED_FALLBACK))


@pytest.fixture(scope="module")
def anthropic_sonnet(proxy: ProxyClient) -> Iterator[Served]:
    yield from _serve(proxy, LiteLLMParamsBody(model=ANTHROPIC_SONNET, api_key="os.environ/ANTHROPIC_API_KEY"))


@pytest.fixture(scope="module")
def bedrock_kimi(proxy: ProxyClient) -> Iterator[Served]:
    yield from _serve(proxy, _bedrock(BEDROCK_KIMI))


@pytest.fixture(scope="module")
def bedrock_nova(proxy: ProxyClient) -> Iterator[Served]:
    yield from _serve(proxy, _bedrock(BEDROCK_NOVA))


@pytest.fixture(scope="module")
def openai_gpt(proxy: ProxyClient) -> Iterator[Served]:
    yield from _serve(proxy, LiteLLMParamsBody(model=OPENAI_GPT, api_key="os.environ/OPENAI_API_KEY"))


def _pdf(pages: int) -> str:
    buffer: Final = io.BytesIO()
    canvas: Final = Canvas(buffer, pagesize=LETTER, invariant=True)
    for page in range(1, pages + 1):
        for line in range(1, PDF_LINES_PER_PAGE + 1):
            canvas.drawString(
                72, 740 - 14 * line, f"Page {page}, line {line}: revenue, costs and headcount for region {line}."
            )
        canvas.showPage()
    canvas.save()
    return base64.b64encode(buffer.getvalue()).decode()


def _user_turn(shape: PromptShape) -> MessageParam:
    question: Final[TextBlockParam] = {"type": "text", "text": f"Reply with one word. {unique_marker()}"}
    if shape.pdf_pages == 0:
        return {"role": "user", "content": [question]}
    document: Final[DocumentBlockParam] = {
        "type": "document",
        "source": {"type": "base64", "media_type": "application/pdf", "data": _pdf(shape.pdf_pages)},
    }
    return {"role": "user", "content": [document, question]}


def _count_and_bill(client: Anthropic, model: str, shape: PromptShape) -> tuple[CountTokensResponse, int]:
    messages: Final = [_user_turn(shape)]
    system: Final[str | Omit] = omit if shape.system is None else shape.system
    tools: Final[list[ToolParam] | Omit] = omit if shape.tool is None else [shape.tool]
    counted: Final = CountTokensResponse.model_validate_json(
        client.messages.with_raw_response.count_tokens(model=model, messages=messages, system=system, tools=tools).text
    )
    usage: Final = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=messages,
        system=system,
        tools=tools,
        extra_body=NO_PROXY_CACHE,
    ).usage
    billed: Final = usage.input_tokens + (usage.cache_creation_input_tokens or 0) + (usage.cache_read_input_tokens or 0)
    return counted, billed


def _within_tolerance(counted: int, billed: int) -> bool:
    return abs(counted - billed) <= billed * COUNT_TOLERANCE


class TestBedrockClaudeCountsWithCostMapFallback:
    @pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.COUNT_TOKENS,
            providers=(Provider.BEDROCK,),
            models=(BEDROCK_SONNET, COST_MAP_FALLBACK),
            capabilities=(Capability.FUNCTION_CALLING, Capability.PDF_INPUT),
            mode=Mode.NONSTREAM,
        )
    )
    def test_count_names_the_fallback_model_and_matches_billed_input(
        self, bedrock_sonnet: Served, sdk: SdkClients, shape: PromptShape
    ) -> None:
        counted, billed = _count_and_bill(sdk.anthropic(bedrock_sonnet.key), bedrock_sonnet.model, shape)

        assert counted.litellm_count_source == "fallback_model", counted
        assert counted.litellm_count_fallback_model == COST_MAP_FALLBACK, counted
        assert _within_tolerance(counted.input_tokens, billed), (counted, billed)


class TestAnthropicClaudeCountsWithProvider:
    @pytest.mark.parametrize("shape", SHAPES, ids=SHAPE_IDS)
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.COUNT_TOKENS,
            providers=(Provider.ANTHROPIC,),
            models=(ANTHROPIC_SONNET,),
            capabilities=(Capability.FUNCTION_CALLING, Capability.PDF_INPUT),
            mode=Mode.NONSTREAM,
        )
    )
    def test_count_comes_from_the_provider_and_matches_billed_input(
        self, anthropic_sonnet: Served, sdk: SdkClients, shape: PromptShape
    ) -> None:
        counted, billed = _count_and_bill(sdk.anthropic(anthropic_sonnet.key), anthropic_sonnet.model, shape)

        assert counted.litellm_count_source == "provider", counted
        assert counted.litellm_count_fallback_model is None, counted
        assert _within_tolerance(counted.input_tokens, billed), (counted, billed)


class TestBedrockKimiCountsWithFramedLocalEstimate:
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.COUNT_TOKENS,
            providers=(Provider.BEDROCK,),
            models=(BEDROCK_KIMI,),
            mode=Mode.NONSTREAM,
        )
    )
    def test_baseline_estimate_matches_billed_input(self, bedrock_kimi: Served, sdk: SdkClients) -> None:
        counted, billed = _count_and_bill(sdk.anthropic(bedrock_kimi.key), bedrock_kimi.model, SHAPES[0])

        assert counted.litellm_count_source == "local_estimate", counted
        assert _within_tolerance(counted.input_tokens, billed), (counted, billed)


class TestConfiguredFallbackModelOverridesCostMap:
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.COUNT_TOKENS,
            providers=(Provider.BEDROCK,),
            models=(BEDROCK_SONNET, CONFIGURED_FALLBACK),
        )
    )
    def test_count_names_the_model_info_fallback(
        self, bedrock_sonnet_with_configured_fallback: Served, sdk: SdkClients
    ) -> None:
        served: Final = bedrock_sonnet_with_configured_fallback
        raw: Final = sdk.anthropic(served.key).messages.with_raw_response.count_tokens(
            model=served.model, messages=[_user_turn(SHAPES[0])]
        )
        counted: Final = CountTokensResponse.model_validate_json(raw.text)

        assert counted.litellm_count_source == "fallback_model", counted
        assert counted.litellm_count_fallback_model == CONFIGURED_FALLBACK, counted
        assert counted.input_tokens > 0, counted


class TestProviderCountErrorWithoutFallback:
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.COUNT_TOKENS,
            providers=(Provider.BEDROCK,),
            models=(BEDROCK_NOVA,),
        )
    )
    def test_count_keeps_the_provider_400_instead_of_estimating(self, bedrock_nova: Served, sdk: SdkClients) -> None:
        with pytest.raises(BadRequestError) as rejected:
            _ = sdk.anthropic(bedrock_nova.key).messages.count_tokens(
                model=bedrock_nova.model, messages=[_user_turn(SHAPES[0])]
            )

        assert rejected.value.status_code == 400, rejected.value


class TestCountSourceOnOtherCountEndpoints:
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.COUNT_TOKENS,
            providers=(Provider.OPENAI,),
            models=(OPENAI_GPT,),
        )
    )
    def test_responses_input_tokens_says_the_provider_counted(self, openai_gpt: Served, sdk: SdkClients) -> None:
        raw: Final = sdk.openai(openai_gpt.key).responses.input_tokens.with_raw_response.count(
            model=openai_gpt.model, input=f"Reply with one word. {unique_marker()}"
        )
        counted: Final = ResponsesInputTokensResponse.model_validate_json(raw.text)

        assert counted.litellm_count_source == "provider", counted
        assert counted.input_tokens > 0, counted

    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.COUNT_TOKENS,
            providers=(Provider.BEDROCK,),
            models=(BEDROCK_KIMI,),
        )
    )
    def test_token_counter_says_kimi_was_estimated_locally(self, proxy: ProxyClient, bedrock_kimi: Served) -> None:
        counted: Final = unwrap(
            proxy.token_counter(
                bedrock_kimi.key,
                TokenCounterBody(
                    model=bedrock_kimi.model,
                    messages=[ChatMessage(role="user", content=f"Reply with one word. {unique_marker()}")],
                ),
            )
        )

        assert counted.count_source == "local_estimate", counted
        assert counted.total_tokens > 0, counted
