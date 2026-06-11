"""The injected HTTP port: the only I/O boundary in the package.

The package never creates a client; the seam passes an object satisfying
``HttpPort`` (functional core, imperative shell). A non-2xx response is a
value (``ProviderHttpError``) the seam converts into the provider exception
contract, never an exception inside the package.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from expression import case, tag, tagged_union
from typing_extensions import assert_never

from ..errors import TranslationError
from ..ir import PlainJson


@dataclass(frozen=True)
class Endpoint:
    """Where and how to reach the provider; built by the seam (URL suffixing,
    auth headers, beta headers are v1-side concerns)."""

    url: str
    headers: Mapping[str, str]
    timeout_seconds: float


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    body: PlainJson
    text: str
    headers: Mapping[str, str]


class HttpPort(Protocol):
    async def post_json(
        self, endpoint: Endpoint, body: Mapping[str, PlainJson]
    ) -> HttpResponse: ...


@dataclass(frozen=True)
class ProviderHttpError:
    """A non-2xx provider response; carries everything the seam needs to
    raise the same exception v1 would."""

    status_code: int
    text: str
    headers: Mapping[str, str]


@tagged_union(frozen=True)
class ExecuteError:
    tag: Literal["translation", "provider_http"] = tag()

    translation: TranslationError = case()
    provider_http: ProviderHttpError = case()

    @staticmethod
    def of_translation(value: TranslationError) -> ExecuteError:
        return ExecuteError(translation=value)

    @staticmethod
    def of_provider_http(value: ProviderHttpError) -> ExecuteError:
        return ExecuteError(provider_http=value)

    @property
    def summary(self) -> str:
        match self.tag:
            case "translation":
                return self.translation.summary
            case "provider_http":
                return f"provider returned HTTP {self.provider_http.status_code}"
        assert_never(self.tag)
