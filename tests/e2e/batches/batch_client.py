"""Client for managed batch-file e2e tests: upload an OpenAI-format batch JSONL
through the proxy and verify the proxy actually persisted a transformed file at the
provider.

The proxy returns a self-describing unified file id (urlsafe-base64) that packs the
target model and the provider's file URI. The tests decode it and read the file
back from the provider, so the guard asserts the real side effect (an ACTIVE,
non-empty file at the provider) rather than just trusting the echoed id - a broken
upload or transform returns a bogus id or fails to persist, and the read-back
fails. Request/response models used only by this suite live here (composition over
the shared Gateway, DI'd in).
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass

from pydantic import BaseModel

import e2e_http
from e2e_gateway import Gateway, build_gateway
from e2e_http import URL, MultipartFile, NoBody, unwrap
from models import KeyGenerateBody

GEMINI_FILES_PREFIX = "https://generativelanguage.googleapis.com/v1beta/files/"


def gemini_api_key() -> str | None:
    """The provider key the read-back needs; tests skip when it is absent."""
    return os.getenv("GEMINI_API_KEY")


class FileUploadForm(BaseModel):
    """The text fields of the /v1/files multipart upload. `target_model_names`
    (comma-separated) routes the file through the managed-files DB path and its
    OpenAI->provider JSONL transform."""

    purpose: str = "batch"
    target_model_names: str


class FileObject(BaseModel):
    id: str
    bytes: int | None = None
    status: str | None = None
    purpose: str | None = None
    object: str | None = None
    filename: str | None = None


class FileDeleteResponse(BaseModel):
    id: str | None = None
    deleted: bool | None = None


class ProviderKeyParam(BaseModel):
    key: str


class ProviderFile(BaseModel):
    """The provider-side (Google AI Studio Files API) view of the uploaded file.
    `sizeBytes` comes back as a string, hence the `size_bytes` accessor."""

    name: str | None = None
    state: str | None = None
    sizeBytes: str | None = None
    mimeType: str | None = None

    @property
    def size_bytes(self) -> int:
        return int(self.sizeBytes) if self.sizeBytes is not None else 0


@dataclass(frozen=True, slots=True)
class UnifiedFileId:
    """The fields the proxy packs into its base64 unified file id."""

    target_models: tuple[str, ...]
    provider_file_uri: str


def parse_unified_file_id(file_id: str) -> UnifiedFileId:
    """Decode the proxy's unified file id: urlsafe-base64 of a ';'-separated list of
    'key,value' pairs, e.g.
    'litellm_proxy:application/jsonl;...;target_model_names,gemini-2.5-flash;llm_output_file_id,https://.../files/abc'.
    Raises if the id is not a managed unified id, so a bogus/echoed id fails the
    test instead of silently passing."""
    padded = file_id + "=" * (-len(file_id) % 4)
    decoded = base64.urlsafe_b64decode(padded).decode("utf-8")
    if not decoded.startswith("litellm_proxy:"):
        raise ValueError(f"not a managed unified file id: {decoded[:60]!r}")
    fields = dict(part.split(",", 1) for part in decoded.split(";") if "," in part)
    uri = fields.get("llm_output_file_id", "")
    if not uri:
        raise ValueError(f"unified id has no provider file uri: {decoded[:80]!r}")
    target = fields.get("target_model_names", "")
    return UnifiedFileId(
        target_models=tuple(model for model in target.split(",") if model),
        provider_file_uri=uri,
    )


def batch_jsonl(model: str, *, lines: int, pad_bytes: int = 0) -> bytes:
    """An OpenAI-format batch input file targeting `model`: `lines` chat-completion
    requests, each optionally padded so the file reaches a chosen size. custom_id is
    unique per line, so a transform that drops or merges lines is detectable."""
    pad = "x" * pad_bytes
    rows = (
        json.dumps(
            {
                "custom_id": f"req-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [{"role": "user", "content": f"{pad}{i}"}],
                    "max_tokens": 4,
                },
            }
        )
        for i in range(lines)
    )
    return ("\n".join(rows) + "\n").encode("utf-8")


@dataclass(frozen=True, slots=True)
class BatchFilesClient:
    gateway: Gateway

    # ---- generic key ops (satisfy lifecycle.ResourceClient via the Gateway) ----

    def generate_key(self, *, models: list[str] | None = None) -> str:
        return self.gateway.generate_key(KeyGenerateBody(models=models or []))

    def delete_key(self, key: str) -> None:
        self.gateway.delete_key(key)

    def delete_customers(self, user_ids: list[str]) -> None:
        self.gateway.delete_customers(user_ids)

    # ---- managed batch files -------------------------------------------

    def upload_batch_file(
        self, key: str, content: bytes, *, target_model: str
    ) -> FileObject:
        return unwrap(
            self.gateway.transport.upload(
                "/v1/files",
                headers=self.gateway.transport.bearer(key),
                form=FileUploadForm(target_model_names=target_model),
                file=MultipartFile(
                    filename="batch.jsonl",
                    content=content,
                    content_type="application/jsonl",
                ),
                response_type=FileObject,
            )
        )

    def delete_file(self, file_id: str) -> None:
        _ = self.gateway.transport.delete(
            f"/v1/files/{file_id}",
            headers=self.gateway.transport.master,
            json=NoBody(),
            response_type=FileDeleteResponse,
        )

    def provider_file(self, uri: str, *, api_key: str) -> ProviderFile:
        """Read the uploaded file back from the provider's Files API. Goes straight
        to the provider URL (not the proxy base), so it proves the proxy really
        created the file rather than echoing an id."""
        return unwrap(
            e2e_http.get(
                URL(uri),
                headers=NoBody(),
                params=ProviderKeyParam(key=api_key),
                response_type=ProviderFile,
            )
        )


def build_client() -> BatchFilesClient:
    return BatchFilesClient(gateway=build_gateway())
