from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass
from typing import Final

from pydantic import JsonValue, TypeAdapter

from ...shared.parity.fixtures.store import read_fixture
from ...shared.parity.recorded_http import RecordedHttpResponse
from ..e2e_parity.sdk.ocr.fixtures.config import DEFAULT_FIXTURE_DIRECTORY
from ..e2e_parity.sdk.ocr.fixtures.models import OcrParityCase
from .models import Profile

SEED: Final = (
    DEFAULT_FIXTURE_DIRECTORY / "mistral-ocr/7727f65058eebe0c68c2a9be97c4777f9a19a7c5a860f5953037e19690bc1154.yaml"
)
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
JSON_PAGES: Final = TypeAdapter(tuple[dict[str, JsonValue], ...])


@dataclass(frozen=True, slots=True)
class Workload:
    profile: Profile
    model: str
    document_url: str
    document_bytes: int
    response: bytes
    response_pages: int
    fixture_sha256: str


def profile_sizes(profile: Profile) -> tuple[int, int]:
    match profile:
        case "small":
            return 32 * 1024, 1
        case "request_medium":
            return 256 * 1024, 1
        case "request_large":
            return 2 * 1024 * 1024, 1
        case "response_medium":
            return 32 * 1024, 16
        case "response_large":
            return 32 * 1024, 128


def padded_pdf(document: bytes, size: int) -> bytes:
    prefix, marker, suffix = document.rpartition(b"startxref")
    if (
        not marker
        or not suffix.rstrip().endswith(b"%%EOF")
        or not document.startswith(b"%PDF-")
        or size < len(document) + 3
    ):
        raise ValueError("expected a PDF seed smaller than the requested document size")
    return prefix + b"%" + b"x" * (size - len(document) - 2) + b"\n" + marker + suffix


def ocr_workload(profile: Profile) -> Workload:
    seed: Final = read_fixture(SEED, OcrParityCase)
    document: Final = seed.litellm_input.document
    response: Final = seed.provider_responses[0]
    if document.type != "document_url" or not isinstance(response, RecordedHttpResponse):
        raise ValueError("OCR benchmark seed must contain an inline PDF and a non-streaming response")
    if not document.document_url.startswith("data:application/pdf;base64,") or response.status_code != 200:
        raise ValueError("OCR benchmark seed must be a successful inline PDF recording")
    document_size, page_count = profile_sizes(profile)
    pdf: Final = padded_pdf(base64.b64decode(document.document_url.split(",", 1)[1], validate=True), document_size)
    body: Final = JSON_OBJECT.validate_json(response.body_bytes())
    pages: Final = JSON_PAGES.validate_python(body["pages"])
    usage: Final = JSON_OBJECT.validate_python(body["usage_info"])
    scaled: Final = {
        **body,
        "pages": tuple({**pages[index % len(pages)], "index": index} for index in range(page_count)),
        "usage_info": {**usage, "pages_processed": page_count, "doc_size_bytes": len(pdf)},
    }
    return Workload(
        profile=profile,
        model=seed.litellm_input.model,
        document_url="data:application/pdf;base64," + base64.b64encode(pdf).decode("ascii"),
        document_bytes=len(pdf),
        response=json.dumps(scaled, separators=(",", ":"), ensure_ascii=False).encode(),
        response_pages=page_count,
        fixture_sha256=hashlib.sha256(SEED.read_bytes()).hexdigest(),
    )
