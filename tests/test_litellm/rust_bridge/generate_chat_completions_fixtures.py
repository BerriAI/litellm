from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Final, cast
from unittest.mock import patch

from dotenv import load_dotenv
from hypothesis import strategies as st

import litellm
from tests.route_parity.fixture_generator import (
    FixtureTarget,
    generate_target_fixtures,
    parse_generator_args,
)
from tests.route_parity.fixture_generator import require_targets as require_fixture_targets
from tests.route_parity.fixture_recorder import ProviderSpec, fixture_directory
from tests.test_litellm.rust_bridge.chat_completions_fixture_models import (
    AnthropicChatCompletionSdkInput,
    ChatCompletionParityCase,
    ChatMessage,
)

FIXTURE_DIR_ENV: Final = "LITELLM_CHAT_COMPLETIONS_FIXTURE_DIR"
_MODEL: Final = "anthropic/claude-sonnet-4-6"

ChatCompletionFixtureTarget = FixtureTarget[AnthropicChatCompletionSdkInput]


def _anthropic_upstream_base(environ: Mapping[str, str]) -> str:
    configured: Final = environ.get("ANTHROPIC_API_BASE", "https://api.anthropic.com").rstrip("/")
    return configured.removesuffix("/v1/messages").removesuffix("/v1")


def _anthropic_target(
    environ: Mapping[str, str],
    sdk_call: Callable[..., object],
) -> ChatCompletionFixtureTarget | None:
    api_key: Final = environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    def invoke(api_base: str, case_input: AnthropicChatCompletionSdkInput) -> object:
        response: Final = sdk_call(api_base=api_base, api_key=api_key, **case_input.as_sdk_kwargs())
        if case_input.stream:
            tuple(cast(Iterable[object], response))
        return response

    return ChatCompletionFixtureTarget(
        name="anthropic-chat-completions",
        provider_spec=ProviderSpec(upstream_base=_anthropic_upstream_base(environ)),
        strategy=st.sampled_from(
            tuple(
                AnthropicChatCompletionSdkInput(
                    model=_MODEL,
                    messages=(ChatMessage(role="user", content="Reply with exactly: parity works"),),
                    max_tokens=16,
                    stream=stream,
                )
                for stream in (False, True)
            )
        ),
        invoke=invoke,
    )


def discover_targets(
    environ: Mapping[str, str],
    sdk_call: Callable[..., object],
) -> tuple[ChatCompletionFixtureTarget, ...]:
    target: Final = _anthropic_target(environ, sdk_call)
    return () if target is None else (target,)


def require_targets(
    targets: tuple[ChatCompletionFixtureTarget, ...],
) -> tuple[ChatCompletionFixtureTarget, ...]:
    return require_fixture_targets(
        targets,
        "No chat completion fixture providers are configured. Set ANTHROPIC_API_KEY",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv()
    args: Final = parse_generator_args()
    sdk_call: Final = cast(Callable[..., object], litellm.completion)
    targets: Final = require_targets(discover_targets(os.environ, sdk_call))
    root: Final = fixture_directory(
        args.fixture_dir,
        os.environ.get(FIXTURE_DIR_ENV),
        Path(__file__).with_name("chat_completions_fixtures"),
    )
    with patch.dict(os.environ, {"LITELLM_RUST": "0"}):
        for target in targets:
            generate_target_fixtures(target, root, args.examples, args.concurrency, ChatCompletionParityCase)


if __name__ == "__main__":
    main()
