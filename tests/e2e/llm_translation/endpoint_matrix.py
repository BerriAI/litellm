"""Provider catalog behind test_endpoint_matrix_e2e.py: one row per route, selectable by env var."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, assert_never

from coverage_registry.schema import LlmRoute
from models import LiteLLMParamsBody

type MatrixEndpoint = Literal[
    "chat_completions",
    "completions",
    "messages",
    "responses",
    "embeddings",
    "images_generations",
    "images_edits",
    "audio_speech",
    "audio_transcriptions",
    "moderations",
]
type Streaming = Literal["stream", "nonstream"]
type AuthMode = Literal["env_ref", "inline", "stored_credential"]

AUTH_MODES: Final[tuple[AuthMode, ...]] = ("env_ref", "inline", "stored_credential")


@dataclass(frozen=True, slots=True)
class Backend:
    model: str
    params: Mapping[str, str] = MappingProxyType({})
    id_route: str | None = None


@dataclass(frozen=True, slots=True)
class Provider:
    route: LlmRoute
    credential: Mapping[str, str]
    backends: Mapping[MatrixEndpoint, Backend]
    edge_mount: str | None = None
    edge_path: str = ""

    def id_route(self, endpoint: MatrixEndpoint) -> str:
        return self.backends[endpoint].id_route or self.route

    def replayable(self) -> bool:
        return self.edge_mount is not None


_ANTHROPIC_HAIKU: Final = "claude-haiku-4-5"
_BEDROCK_HAIKU: Final = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
_GEMINI_FLASH: Final = "gemini-3.8-flash"
_TOGETHER_LLAMA: Final = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo"
_VERTEX_GLOBAL: Final[Mapping[str, str]] = MappingProxyType({"vertex_location": "global"})
_VERTEX_US_CENTRAL: Final[Mapping[str, str]] = MappingProxyType({"vertex_location": "us-central1"})
_BEDROCK_REGION: Final[Mapping[str, str]] = MappingProxyType({"aws_region_name": "us-east-1"})

PROVIDERS: Final[tuple[Provider, ...]] = (
    Provider(
        route="openai",
        credential=MappingProxyType({"api_key": "OPENAI_API_KEY"}),
        edge_mount="openai",
        edge_path="/v1",
        backends=MappingProxyType(
            {
                "chat_completions": Backend("openai/gpt-5.4-mini"),
                "completions": Backend("openai/gpt-3.5-turbo-instruct"),
                "messages": Backend("openai/gpt-5.4-mini"),
                "responses": Backend("openai/gpt-5.4-mini"),
                "embeddings": Backend("openai/text-embedding-3-small"),
                "images_generations": Backend("openai/gpt-image-1-mini"),
                "images_edits": Backend("openai/gpt-image-1-mini"),
                "audio_speech": Backend("openai/gpt-4o-mini-tts"),
                "audio_transcriptions": Backend("openai/gpt-4o-mini-transcribe"),
                "moderations": Backend("openai/omni-moderation-latest"),
            }
        ),
    ),
    Provider(
        route="anthropic",
        credential=MappingProxyType({"api_key": "ANTHROPIC_API_KEY"}),
        edge_mount="anthropic",
        backends=MappingProxyType(
            {
                "chat_completions": Backend(f"anthropic/{_ANTHROPIC_HAIKU}"),
                "messages": Backend(f"anthropic/{_ANTHROPIC_HAIKU}"),
                "responses": Backend(f"anthropic/{_ANTHROPIC_HAIKU}"),
            }
        ),
    ),
    Provider(
        route="gemini",
        credential=MappingProxyType({"api_key": "GEMINI_API_KEY"}),
        backends=MappingProxyType(
            {
                "chat_completions": Backend(f"gemini/{_GEMINI_FLASH}"),
                "messages": Backend(f"gemini/{_GEMINI_FLASH}"),
                "responses": Backend(f"gemini/{_GEMINI_FLASH}"),
                "embeddings": Backend("gemini/gemini-embedding-001"),
            }
        ),
    ),
    Provider(
        route="vertex",
        credential=MappingProxyType(
            {"vertex_credentials": "VERTEXAI_CREDENTIALS", "vertex_project": "VERTEXAI_PROJECT"}
        ),
        backends=MappingProxyType(
            {
                "chat_completions": Backend(f"vertex_ai/{_GEMINI_FLASH}", _VERTEX_GLOBAL),
                "messages": Backend(f"vertex_ai/{_GEMINI_FLASH}", _VERTEX_GLOBAL),
                "responses": Backend(f"vertex_ai/{_GEMINI_FLASH}", _VERTEX_GLOBAL),
                "embeddings": Backend("vertex_ai/text-embedding-005", _VERTEX_US_CENTRAL),
            }
        ),
    ),
    Provider(
        route="bedrock_converse",
        credential=MappingProxyType({"api_key": "AWS_BEARER_TOKEN_BEDROCK"}),
        backends=MappingProxyType(
            {
                "chat_completions": Backend(_BEDROCK_HAIKU, _BEDROCK_REGION),
                "messages": Backend(_BEDROCK_HAIKU, _BEDROCK_REGION),
                "responses": Backend(_BEDROCK_HAIKU, _BEDROCK_REGION),
                "embeddings": Backend("bedrock/amazon.titan-embed-text-v2:0", _BEDROCK_REGION, id_route="bedrock"),
            }
        ),
    ),
    Provider(
        route="together_ai",
        credential=MappingProxyType({"api_key": "TOGETHER_API_KEY"}),
        backends=MappingProxyType(
            {
                "chat_completions": Backend(_TOGETHER_LLAMA),
                "messages": Backend(_TOGETHER_LLAMA),
            }
        ),
    ),
)


def selected_providers() -> tuple[Provider, ...]:
    by_route: Final = {provider.route: provider for provider in PROVIDERS}
    return tuple(by_route[route] for route in _selection("E2E_MATRIX_PROVIDERS", tuple(by_route)))


def selected_auth_modes() -> tuple[AuthMode, ...]:
    chosen: Final = _selection("E2E_MATRIX_AUTH_MODES", AUTH_MODES)
    return tuple(mode for mode in AUTH_MODES if mode in chosen)


def _selection(variable: str, known: tuple[str, ...]) -> tuple[str, ...]:
    raw: Final = os.environ.get(variable, "").strip()
    if not raw:
        return known
    chosen: Final = tuple(name.strip() for name in raw.split(",") if name.strip())
    unknown: Final = tuple(name for name in chosen if name not in known)
    if unknown:
        raise ValueError(f"{variable} names unknown entries {unknown}; known: {', '.join(known)}")
    return chosen


def missing_credentials(provider: Provider) -> tuple[str, ...]:
    return tuple(env for env in provider.credential.values() if not os.environ.get(env))


def credential_values(provider: Provider) -> Mapping[str, str]:
    missing: Final = missing_credentials(provider)
    if missing:
        raise RuntimeError(f"{provider.route} inline/stored auth needs {', '.join(missing)} in the test environment")
    return MappingProxyType({field: os.environ[env] for field, env in provider.credential.items()})


def deployment_params(
    provider: Provider,
    endpoint: MatrixEndpoint,
    auth_mode: AuthMode,
    *,
    edge_base: str | None,
    credential_name: str | None,
) -> LiteLLMParamsBody:
    backend: Final = provider.backends[endpoint]
    auth: Final = _auth_fields(provider, auth_mode, credential_name)
    api_base: Final = {} if edge_base is None else {"api_base": f"{edge_base}{provider.edge_path}"}
    return LiteLLMParamsBody.model_validate({"model": backend.model, **backend.params, **auth, **api_base})


def _auth_fields(provider: Provider, auth_mode: AuthMode, credential_name: str | None) -> Mapping[str, str]:
    match auth_mode:
        case "env_ref":
            return MappingProxyType({field: f"os.environ/{env}" for field, env in provider.credential.items()})
        case "inline":
            return credential_values(provider)
        case "stored_credential":
            if credential_name is None:
                raise ValueError("stored_credential auth needs the name of the credential registered for the case")
            return MappingProxyType({"litellm_credential_name": credential_name})
        case _:
            assert_never(auth_mode)
