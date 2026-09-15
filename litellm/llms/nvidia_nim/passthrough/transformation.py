from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Final

import httpx

from litellm.llms.base_llm.passthrough.transformation import (
    BasePassthroughConfig,
    model_group_from,
    relayed_body,
    strip_leading_model_segment,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import StandardPassThroughResponseObject

if TYPE_CHECKING:
    from httpx import URL, Response

    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.llms.base_llm.ocr.transformation import OCRResponse
    from litellm.llms.base_llm.passthrough.transformation import LoggedRelayResponse


API_VERSION_SEGMENT: Final = re.compile(r"^v\d+$")


def nvidia_nim_router_model_in_endpoint(endpoint: str, router_models: Collection[str]) -> str | None:
    segments: Final = tuple(segment for segment in endpoint.split("/") if segment)
    return next(
        (
            "/".join(segments[:length])
            for length in range(len(segments), 0, -1)
            if "/".join(segments[:length]) in router_models
        ),
        None,
    )


def without_repeated_version_prefix(api_base: str, native_endpoint: str) -> str:
    url: Final = httpx.URL(api_base)
    base_segments: Final = tuple(segment for segment in url.path.split("/") if segment)
    first_native_segment: Final = native_endpoint.lstrip("/").split("/", 1)[0]
    repeated: Final = (
        bool(base_segments)
        and API_VERSION_SEGMENT.match(first_native_segment) is not None
        and base_segments[-1] == first_native_segment
    )
    kept_segments: Final = base_segments[:-1] if repeated else base_segments
    return str(url.copy_with(path="/" + "/".join(kept_segments), query=None)).rstrip("/")


class NvidiaNimPassthroughConfig(BasePassthroughConfig):
    def is_streaming_request(self, endpoint: str, request_data: dict) -> bool:
        return bool(request_data.get("stream", False))

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        endpoint: str,
        request_query_params: dict | None,
        litellm_params: dict,
    ) -> tuple[URL, str]:
        base_target_url: Final = self.get_api_base(api_base)
        if base_target_url is None:
            raise ValueError("NVIDIA NIM api base not found: set `api_base` on the deployment or NVIDIA_NIM_API_BASE")
        native_endpoint: Final = strip_leading_model_segment(endpoint, (model_group_from(litellm_params), model))
        root: Final = without_repeated_version_prefix(base_target_url, native_endpoint)
        return (self.format_url(native_endpoint, root, request_query_params), root)

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: base class contract returns dict for httpx
        if api_key is None:
            return dict(headers)  # mutable-ok: base class contract returns dict for httpx
        return {
            **headers,
            "Authorization": f"Bearer {api_key}",
        }  # mutable-ok: base class contract returns dict for httpx

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str | None:
        return api_base or get_secret_str("NVIDIA_NIM_API_BASE")

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key or get_secret_str("NVIDIA_NIM_API_KEY")

    @staticmethod
    def get_base_model(model: str) -> str | None:
        return model

    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        return []

    def logging_non_streaming_response(
        self,
        model: str,
        custom_llm_provider: str,
        httpx_response: Response,
        request_data: Mapping[str, object],
        logging_obj: Logging,
        endpoint: str,
    ) -> LoggedRelayResponse | OCRResponse | StandardPassThroughResponseObject | None:
        return StandardPassThroughResponseObject(response=relayed_body(httpx_response))
