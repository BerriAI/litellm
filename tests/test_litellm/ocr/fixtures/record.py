from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

from dotenv import load_dotenv

import litellm
from litellm.rust_bridge.ocr import use_litellm_rust
from tests.route_parity.fixtures.pipeline import parse_recording_args, record_fixtures
from tests.route_parity.fixtures.store import fixture_directory
from tests.test_litellm.ocr.fixtures.azure import (
    azure_document_intelligence_recording_targets,
    azure_mistral_recording_targets,
)
from tests.test_litellm.ocr.fixtures.base import OcrSdkInputBase
from tests.test_litellm.ocr.fixtures.common import OcrFixtureClient, OcrRecordingTarget, OcrSdkCall
from tests.test_litellm.ocr.fixtures.mistral import mistral_recording_targets
from tests.test_litellm.ocr.fixtures.models import OcrParityCase
from tests.test_litellm.ocr.fixtures.reducto import reducto_recording_targets
from tests.test_litellm.ocr.fixtures.vertex import vertex_recording_targets

FIXTURE_DIR_ENV: Final = "LITELLM_OCR_FIXTURE_DIR"


class LiteLLMOcrFixtureClient:
    def __init__(self, sdk_call: OcrSdkCall) -> None:
        self.sdk_call: Final = sdk_call

    def execute(self, api_base: str, api_key: str, case_input: OcrSdkInputBase) -> None:
        self.sdk_call(api_base=api_base, api_key=api_key, **case_input.as_sdk_kwargs())


def discover_targets(
    environ: Mapping[str, str],
    client: OcrFixtureClient,
) -> tuple[OcrRecordingTarget, ...]:
    return (
        *mistral_recording_targets(environ, client),
        *azure_mistral_recording_targets(environ, client),
        *azure_document_intelligence_recording_targets(environ, client),
        *vertex_recording_targets(environ, client),
        *reducto_recording_targets(environ, client),
    )


def require_targets(targets: tuple[OcrRecordingTarget, ...]) -> tuple[OcrRecordingTarget, ...]:
    if targets:
        return targets
    raise SystemExit("No OCR fixture providers are configured. Set a supported provider API key and endpoint")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv()
    args: Final = parse_recording_args()
    client: Final = LiteLLMOcrFixtureClient(cast(OcrSdkCall, litellm.ocr))
    targets: Final = require_targets(discover_targets(os.environ, client))
    root: Final = fixture_directory(
        args.fixture_dir,
        os.environ.get(FIXTURE_DIR_ENV),
        Path(__file__).with_name("data"),
    )
    use_litellm_rust(False, ocr=None, aocr=None)
    summary: Final = record_fixtures(targets, root, args.examples, args.concurrency, OcrParityCase)
    return summary.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
