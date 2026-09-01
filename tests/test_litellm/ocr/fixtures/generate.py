from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Final, cast

from dotenv import load_dotenv

import litellm
from litellm.rust_bridge.ocr import use_litellm_rust
from tests.route_parity.fixture_generator import (
    FixtureProvider,
    FixtureSdkCall,
    discover_fixture_targets,
    generate_target_fixtures,
    parse_generator_args,
)
from tests.route_parity.fixture_generator import require_targets as require_fixture_targets
from tests.route_parity.fixture_recorder import fixture_directory
from tests.test_litellm.ocr.fixtures.azure import (
    AzureDocumentIntelligenceFixtureProvider,
    AzureMistralFixtureProvider,
    azure_document_intelligence_input_strategy,
)
from tests.test_litellm.ocr.fixtures.common import OcrFixtureTarget
from tests.test_litellm.ocr.fixtures.mistral import (
    MistralFixtureProvider,
    mistral_input_strategy,
)
from tests.test_litellm.ocr.fixtures.models import OcrParityCase, OcrSdkInputBase
from tests.test_litellm.ocr.fixtures.reducto import (
    ReductoFixtureProvider,
    reducto_legacy_input_strategy,
    reducto_v3_input_strategy,
)
from tests.test_litellm.ocr.fixtures.vertex import (
    VertexFixtureProvider,
    vertex_deepseek_input_strategy,
)

__all__ = (
    "azure_document_intelligence_input_strategy",
    "mistral_input_strategy",
    "reducto_legacy_input_strategy",
    "reducto_v3_input_strategy",
    "vertex_deepseek_input_strategy",
)

FIXTURE_DIR_ENV: Final = "LITELLM_OCR_FIXTURE_DIR"
OCR_FIXTURE_PROVIDERS: Final[tuple[FixtureProvider[OcrSdkInputBase], ...]] = (
    MistralFixtureProvider(),
    AzureMistralFixtureProvider(),
    AzureDocumentIntelligenceFixtureProvider(),
    VertexFixtureProvider(),
    ReductoFixtureProvider(),
)


def discover_targets(
    environ: Mapping[str, str],
    sdk_call: FixtureSdkCall,
) -> tuple[OcrFixtureTarget, ...]:
    return discover_fixture_targets(OCR_FIXTURE_PROVIDERS, environ, sdk_call)


def require_targets(targets: tuple[OcrFixtureTarget, ...]) -> tuple[OcrFixtureTarget, ...]:
    return require_fixture_targets(
        targets,
        "No OCR fixture providers are configured. Set a supported provider API key and endpoint",
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv()
    args: Final = parse_generator_args()
    sdk_call: Final = cast(FixtureSdkCall, litellm.ocr)
    targets: Final = require_targets(discover_targets(os.environ, sdk_call))
    root: Final = fixture_directory(
        args.fixture_dir,
        os.environ.get(FIXTURE_DIR_ENV),
        Path(__file__).with_name("data"),
    )
    use_litellm_rust(False, ocr=None, aocr=None)
    for target in targets:
        generate_target_fixtures(target, root, args.examples, args.concurrency, OcrParityCase)


if __name__ == "__main__":
    main()
