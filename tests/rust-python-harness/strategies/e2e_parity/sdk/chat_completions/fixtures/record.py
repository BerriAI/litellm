from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from dotenv import load_dotenv
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

import litellm
from litellm.rust_bridge.configuration import use_litellm_rust

from ......shared.parity.fixtures.cli import parse_recording_args
from ......shared.parity.fixtures.pipeline import RecordingTarget, record_fixtures
from ......shared.parity.fixtures.recording import UpstreamEndpoint
from ......shared.parity.fixtures.store import fixture_directory, fixture_id
from ......shared.parity.recorded_http import RecordedResponse
from ..test_sdk_parity import CASES, ChatCase

FIXTURE_DIR_ENV: Final = "LITELLM_CHAT_COMPLETIONS_FIXTURE_DIR"
DEFAULT_FIXTURE_DIRECTORY: Final = Path(__file__).with_name("data")


def input_strategy(model: str) -> SearchStrategy[ChatCase]:
    conversations: Final = st.sampled_from(
        (
            [{"role": "user", "content": "hello"}],
            [
                {"role": "system", "content": "be concise"},
                {"role": "user", "content": "hello"},
            ],
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": [{"type": "text", "text": "follow up"}]},
            ],
        )
    )
    params: Final = st.fixed_dictionaries(
        {"max_tokens": st.integers(min_value=1, max_value=64)},
        optional={
            "temperature": st.just(1.0),
            "top_p": st.just(1.0),
            "stop": st.lists(st.text(min_size=1, max_size=8), min_size=1, max_size=2),
        },
    )
    return st.builds(
        ChatCase,
        name=st.just("generated"),
        model=st.just(model),
        messages=conversations,
        optional_params=params,
        provider_responses=st.just(()),
        expected=st.just("success"),
    )


@dataclass(frozen=True, slots=True)
class ChatInvocation:
    api_key: str

    def execute(self, provider_url: str, case_input: ChatCase) -> None:
        call: Final = cast(Callable[..., object], litellm.completion)
        call(
            model=case_input.model,
            messages=case_input.messages,
            api_base=provider_url,
            api_key=self.api_key,
            num_retries=0,
            **case_input.optional_params,
        )


def _case_factory(case: ChatCase, responses: tuple[RecordedResponse, ...]) -> ChatCase:
    return case.model_copy(update={"name": fixture_id(case, case.model), "provider_responses": responses})


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv()
    args: Final = parse_recording_args()
    api_key: Final = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY to record Chat Completions parity fixtures")
    api_base: Final = os.environ.get("ANTHROPIC_API_BASE", "https://api.anthropic.com")
    target: Final = RecordingTarget(
        name="anthropic-chat-completions",
        upstream=UpstreamEndpoint(base_url=api_base),
        strategy=input_strategy("anthropic/claude-sonnet-5"),
        invocation=ChatInvocation(api_key),
        required_inputs=tuple(case for case in CASES if case.expected == "success"),
    )
    root: Final = fixture_directory(
        args.fixture_dir,
        os.environ.get(FIXTURE_DIR_ENV),
        DEFAULT_FIXTURE_DIRECTORY,
    )
    use_litellm_rust(False)
    summary: Final = record_fixtures((target,), root, args.examples, args.concurrency, ChatCase, _case_factory)
    return summary.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
