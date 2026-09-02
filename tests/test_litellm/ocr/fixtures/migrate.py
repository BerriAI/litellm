from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final, cast

import litellm
from litellm.rust_bridge.ocr import use_litellm_rust
from tests.route_parity.fixtures.recording import (
    RecordedInteraction,
    UpstreamEndpoint,
    record_upstream_interactions,
)
from tests.route_parity.fixtures.store import FixtureEnvelope, read_fixture, save_fixture
from tests.route_parity.replay import replay_server
from tests.test_litellm.ocr.fixtures.common import OcrSdkCall
from tests.test_litellm.ocr.fixtures.config import configured_fixture_directory
from tests.test_litellm.ocr.fixtures.models import OcrParityCase, OcrSdkInput


def _invoke(provider_url: str, case_input: OcrSdkInput) -> object:
    sdk_call: Final = cast(OcrSdkCall, litellm.ocr)
    return sdk_call(api_base=provider_url, api_key="test-key", **case_input.as_sdk_kwargs())


def migrate_fixture(path: Path) -> Path:
    case: Final = read_fixture(path, OcrParityCase)
    envelope: Final = FixtureEnvelope.model_validate_json(path.read_text(encoding="utf-8"))
    with replay_server() as provider:
        for response in case.provider_responses:
            provider.enqueue_response(response)
        captured: Final = record_upstream_interactions(UpstreamEndpoint(provider.url), case.litellm_input, _invoke)
        provider.take_requests(len(case.provider_responses))
    interactions: Final = tuple(
        RecordedInteraction(item.request, response)
        for item, response in zip(captured, case.provider_responses, strict=True)
    )
    destination: Final = save_fixture(
        path.parent,
        case.litellm_input,
        case,
        interactions,
        recorded_at=envelope.recorded_at,
        request_source="python_replay",
    )
    read_fixture(destination, OcrParityCase)
    path.unlink()
    return destination


def main() -> None:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", type=Path, default=configured_fixture_directory())
    args: Final = parser.parse_args()
    directory: Final = cast(Path, args.fixture_dir)
    use_litellm_rust(False, ocr=None, aocr=None)
    paths: Final = tuple(sorted(directory.rglob("*.json")))
    for path in paths:
        print(f"Migrated {path.name} to {migrate_fixture(path).name}")


if __name__ == "__main__":
    main()
