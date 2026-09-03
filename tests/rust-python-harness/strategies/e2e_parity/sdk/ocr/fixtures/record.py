from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Final, cast

from dotenv import load_dotenv

import litellm
from litellm.rust_bridge.ocr import use_litellm_rust
from ......shared.parity.fixtures.cli import parse_recording_args
from ......shared.parity.fixtures.media import structured_image_data_uri
from ......shared.parity.fixtures.pipeline import record_fixtures
from ......shared.parity.fixtures.store import fixture_directory
from .azure import (
    azure_document_intelligence_recording_targets,
    azure_mistral_recording_targets,
)
from .base import OcrSdkInputBase
from .common import OcrFixtureClient, OcrRecordingTarget, OcrSdkCall
from .config import DEFAULT_FIXTURE_DIRECTORY, FIXTURE_DIR_ENV
from .mistral import mistral_recording_targets
from .models import OcrParityCase
from .reducto import reducto_recording_targets
from .vertex import vertex_recording_targets


class LiteLLMOcrFixtureClient:
    def __init__(self, sdk_call: OcrSdkCall) -> None:
        self.sdk_call: Final = sdk_call

    def execute(self, api_base: str, api_key: str, case_input: OcrSdkInputBase) -> None:
        self.sdk_call(api_base=api_base, api_key=api_key, **case_input.as_sdk_kwargs())


def discover_targets(
    environ: Mapping[str, str],
    client: OcrFixtureClient,
    inline_image_data_uri: str,
) -> tuple[OcrRecordingTarget, ...]:
    return (
        *mistral_recording_targets(environ, client, inline_image_data_uri),
        *azure_mistral_recording_targets(environ, client, inline_image_data_uri),
        *azure_document_intelligence_recording_targets(environ, client),
        *vertex_recording_targets(environ, client, inline_image_data_uri),
        *reducto_recording_targets(environ, client, inline_image_data_uri),
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
    inline_image_data_uri: Final = structured_image_data_uri()
    targets: Final = require_targets(discover_targets(os.environ, client, inline_image_data_uri))
    root: Final = fixture_directory(
        args.fixture_dir,
        os.environ.get(FIXTURE_DIR_ENV),
        DEFAULT_FIXTURE_DIRECTORY,
    )
    use_litellm_rust(False, ocr=None, aocr=None)
    summary: Final = record_fixtures(targets, root, args.examples, args.concurrency, OcrParityCase)
    return summary.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
