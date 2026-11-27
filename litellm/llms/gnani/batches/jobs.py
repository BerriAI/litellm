from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PureWindowsPath
from types import MappingProxyType
from typing import BinaryIO, Final, Generic, TypeVar
from uuid import UUID

import httpx
from httpx._types import RequestFiles
from pydantic import JsonValue

from litellm.constants import GNANI_BATCH_API_PATH
from litellm.exceptions import AuthenticationError
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.secret_managers.main import get_secret_str

_Result = TypeVar("_Result")


def batch_api_base(api_base: str | None) -> str:
    return join_api_base(api_base or get_secret_str("GNANI_API_BASE") or "https://api.vachana.ai", GNANI_BATCH_API_PATH)


def _api_key(key: str | None) -> str:
    resolved: Final = key or get_secret_str("GNANI_API_KEY")
    if not resolved:
        raise AuthenticationError(message="Set GNANI_API_KEY or pass api_key", llm_provider="gnani", model="")
    return resolved


def _page(status: str | None, limit: int | None, cursor: str | None) -> Mapping[str, str | int]:
    fields: Final = {"status": status, "limit": limit, "cursor": cursor}
    return {name: value for name, value in fields.items() if value is not None}


@dataclass(frozen=True, slots=True)
class GnaniBatchJobs(Generic[_Result]):
    _base: str
    _key: str = field(repr=False)
    _send: Callable[[httpx.Request], _Result] = field(repr=False)

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, object] | None = None,
        parts: RequestFiles | None = None,
        params: Mapping[str, str | int] | None = None,
    ) -> _Result:
        return self._send(
            httpx.Request(
                method,
                f"{self._base}/{path}",
                headers={"X-API-Key-ID": self._key},
                json=body,
                files=parts,
                params=params,
            )
        )

    def create(
        self,
        *,
        model: str,
        language_code: str,
        urls: Sequence[str] = (),
        files: Sequence[tuple[str, BinaryIO | bytes]] = (),
        upload_filenames: Sequence[str] = (),
        with_diarization: bool | None = None,
        num_speakers: int | None = None,
        callback_url: str | None = None,
        options: Mapping[str, JsonValue] = MappingProxyType({}),
    ) -> _Result:
        """Create a job from public audio URLs, local files, or filenames for presigned PUTs. Call start after uploading."""
        if sum(map(bool, (urls, files, upload_filenames))) != 1:
            raise ValueError("Supply exactly one of urls, files or upload_filenames")
        config: Final = {
            **options,
            "model": model.removeprefix("gnani/"),
            "language_code": language_code,
            **({"with_diarization": with_diarization} if with_diarization is not None else {}),
            **({"num_speakers": num_speakers} if num_speakers is not None else {}),
        }
        if files:
            callback: Final = (("callback_url", (None, callback_url)),) if callback_url is not None else ()
            parts: Final[RequestFiles] = (
                ("config", (None, json.dumps(config), "application/json")),
                *(("files", (PureWindowsPath(name).name, audio)) for name, audio in files),
                *callback,
            )
            return self._request("POST", "jobs", parts=parts)
        names: Final = (upload_filenames,) if isinstance(upload_filenames, str) else upload_filenames
        paths: Final = (urls,) if isinstance(urls, str) else urls
        source: Final = (
            {
                "type": "direct_upload_url",
                "files": [{"filename": PureWindowsPath(name).name} for name in names],
            }
            if upload_filenames
            else {"type": "cloud_storage", "auth": {"mode": "public"}, "paths": list(paths)}
        )
        body: Final = {"config": config, "source": source, **({"callback_url": callback_url} if callback_url else {})}
        return self._request("POST", "v2/jobs" if upload_filenames else "jobs", body=body)

    def start(self, job_id: str) -> _Result:
        return self._request("POST", f"jobs/{UUID(job_id)}/start")

    def get(self, job_id: str) -> _Result:
        return self._request("GET", f"jobs/{UUID(job_id)}")

    def list(self, *, status: str | None = None, limit: int | None = None, cursor: str | None = None) -> _Result:
        return self._request("GET", "jobs", params=_page(status, limit, cursor))

    def files(
        self, job_id: str, *, status: str | None = None, limit: int | None = None, cursor: str | None = None
    ) -> _Result:
        return self._request("GET", f"jobs/{UUID(job_id)}/files", params=_page(status, limit, cursor))

    def cancel(self, job_id: str, *, reason: str | None = None) -> _Result:
        return self._request(
            "POST", f"jobs/{UUID(job_id)}/cancel", body={"reason": reason} if reason is not None else None
        )


def gnani_batch(
    *, api_key: str | None = None, api_base: str | None = None, client: httpx.Client | None = None
) -> GnaniBatchJobs[httpx.Response]:
    """Manage Gnani Prisma audio transcription jobs with a synchronous HTTP client."""
    from litellm.llms.custom_httpx.http_handler import (
        get_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # the shared pool accepts untyped optional settings
    )

    transport: Final = client if client is not None else get_httpx_client().client

    def send(request: httpx.Request) -> httpx.Response:
        return transport.send(request, follow_redirects=False).raise_for_status()

    return GnaniBatchJobs(batch_api_base(api_base), _api_key(api_key), send)


def agnani_batch(
    *, api_key: str | None = None, api_base: str | None = None, client: httpx.AsyncClient | None = None
) -> GnaniBatchJobs[Awaitable[httpx.Response]]:
    """Manage the same jobs with awaitable operations and a pooled async HTTP client."""
    from litellm.llms.custom_httpx.http_handler import (
        get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # the shared pool accepts untyped optional settings
    )

    transport: Final = client if client is not None else get_async_httpx_client(llm_provider="gnani").client

    async def send(request: httpx.Request) -> httpx.Response:
        return (await transport.send(request, follow_redirects=False)).raise_for_status()

    return GnaniBatchJobs(batch_api_base(api_base), _api_key(api_key), send)
