"""The credentials are `os.environ/` references the gateway resolves, the same names the Claude Code matrix's
`test_config.yaml` uses, and Bedrock rides the gateway's default AWS identity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, Literal

import pytest
from _pytest.mark.structures import ParameterSet
from cli_harness.rate_limiter import RateBucket
from codex.harness import CodexCatalog
from codex.run import CodexRun
from e2e_metadata import Domain, Mode, Provider, Route, Subject, meta
from models import LiteLLMParamsBody

GPT: Final = "gpt-6.1-sol"
CLAUDE: Final = "claude-sonnet-5-5"
GPT_ON_BEDROCK_MANTLE: Final = f"bedrock_mantle/openai.{GPT}"
GPT_ON_BEDROCK_RUNTIME: Final = f"bedrock/global.openai.{GPT}"
GPT_ON_AZURE_OPENAI: Final = f"azure/{GPT}"
GPT_ON_OPENAI: Final = f"openai/{GPT}"
CLAUDE_ON_BEDROCK: Final = f"bedrock/us.anthropic.{CLAUDE}"
_BEDROCK_REGION: Final = "us-east-1"

type SetupId = Literal["bedrock_mantle", "bedrock_runtime", "azure_openai", "openai", "claude_bedrock", "vertex_ai"]


@dataclass(frozen=True, slots=True)
class CodexEntry:
    catalog: CodexCatalog
    slug: str


@dataclass(frozen=True, slots=True)
class Deployment:
    litellm_params: LiteLLMParamsBody
    rate_bucket: RateBucket
    real_model_name: str
    codex_entry: CodexEntry


@dataclass(frozen=True, slots=True)
class ModelSetup:
    model: str
    rate_bucket: RateBucket
    api_key: str = field(repr=False)


def _drives(provider: Provider, model: str) -> pytest.MarkDecorator:
    return meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.RESPONSES,
            providers=(provider,),
            models=(model,),
            mode=Mode.STREAM,
        )
    )


GEMINI_ON_VERTEX: Final = "vertex_ai/gemini-4-neon"
_VERTEX_LOCATION: Final = "global"

SETUPS: Final = (
    pytest.param(
        Deployment(
            LiteLLMParamsBody(model=GPT_ON_BEDROCK_MANTLE, aws_region_name=_BEDROCK_REGION),
            RateBucket.BEDROCK_MANTLE,
            real_model_name=GPT,
            codex_entry=CodexEntry("amazon-bedrock", GPT_ON_BEDROCK_MANTLE.removeprefix("bedrock_mantle/")),
        ),
        id="bedrock_mantle",
        marks=_drives(Provider.BEDROCK_MANTLE, GPT_ON_BEDROCK_MANTLE),
    ),
    pytest.param(
        Deployment(
            LiteLLMParamsBody(model=GPT_ON_BEDROCK_RUNTIME, aws_region_name=_BEDROCK_REGION),
            RateBucket.BEDROCK_CONVERSE,
            real_model_name=GPT,
            codex_entry=CodexEntry("amazon-bedrock-runtime", GPT_ON_BEDROCK_RUNTIME.removeprefix("bedrock/")),
        ),
        id="bedrock_runtime",
        marks=(
            _drives(Provider.BEDROCK, GPT_ON_BEDROCK_RUNTIME),
            pytest.mark.skip(reason="Stage IAM lacks bedrock:InvokeModel on the default Bedrock Runtime project"),
        ),
    ),
    pytest.param(
        Deployment(
            LiteLLMParamsBody(
                model=GPT_ON_AZURE_OPENAI, api_base="os.environ/AZURE_API_BASE", api_key="os.environ/AZURE_API_KEY"
            ),
            RateBucket.AZURE_OPENAI,
            real_model_name=GPT,
            codex_entry=CodexEntry("openai", GPT),
        ),
        id="azure_openai",
        marks=_drives(Provider.AZURE, GPT_ON_AZURE_OPENAI),
    ),
    pytest.param(
        Deployment(
            LiteLLMParamsBody(model=GPT_ON_OPENAI, api_key="os.environ/OPENAI_API_KEY"),
            RateBucket.OPENAI,
            real_model_name=GPT,
            codex_entry=CodexEntry("openai", GPT),
        ),
        id="openai",
        marks=_drives(Provider.OPENAI, GPT_ON_OPENAI),
    ),
    pytest.param(
        Deployment(
            LiteLLMParamsBody(model=CLAUDE_ON_BEDROCK, aws_region_name=_BEDROCK_REGION),
            RateBucket.BEDROCK_CONVERSE,
            real_model_name=CLAUDE,
            codex_entry=CodexEntry("openai", CLAUDE),
        ),
        id="claude_bedrock",
        marks=_drives(Provider.BEDROCK, CLAUDE_ON_BEDROCK),
    ),
    pytest.param(
        Deployment(
            LiteLLMParamsBody(
                model=GEMINI_ON_VERTEX,
                vertex_project="os.environ/VERTEXAI_PROJECT",
                vertex_location=_VERTEX_LOCATION,
                vertex_credentials="os.environ/VERTEXAI_CREDENTIALS",
            ),
            RateBucket.VERTEX_AI,
            real_model_name=GEMINI_ON_VERTEX.removeprefix("vertex_ai/"),
            codex_entry=CodexEntry("openai", GEMINI_ON_VERTEX.removeprefix("vertex_ai/")),
        ),
        id="vertex_ai",
        marks=_drives(Provider.VERTEX_AI, GEMINI_ON_VERTEX),
    ),
)


class MissingFeature(AssertionError):
    pass


def require_feature(present: bool, run: CodexRun) -> None:
    if not present:
        raise MissingFeature(run)


def with_expected_failures(reasons: Mapping[SetupId, str]) -> tuple[ParameterSet, ...]:
    return tuple(_expected_to_fail(setup, reasons) for setup in SETUPS)


def _expected_to_fail(setup: ParameterSet, reasons: Mapping[SetupId, str]) -> ParameterSet:
    reason: Final = next((reason for setup_id, reason in reasons.items() if setup_id == setup.id), None)
    if reason is None:
        return setup
    expected_failure: Final = pytest.mark.xfail(strict=True, raises=MissingFeature, reason=reason)
    return pytest.param(*setup.values, id=setup.id, marks=(*setup.marks, expected_failure))
