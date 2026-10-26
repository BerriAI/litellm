from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Final, Literal

from pydantic import ConfigDict, Field, field_validator
from typing_extensions import TypedDict

from litellm.types.llms.base import LiteLLMBaseModel

OPENROUTER_REASONING_EFFORTS: Final = ("max", "xhigh", "high", "medium", "low", "minimal", "none")


class OpenRouterErrorMessage(TypedDict):
    message: str
    code: int
    metadata: dict


class OpenRouterProviderPolicy(LiteLLMBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["openrouter"]
    only: Annotated[tuple[Annotated[str, Field(min_length=1, strict=True)], ...], Field(min_length=1)]
    allow_fallbacks: Literal[False] = False

    @field_validator("only")
    @classmethod
    def exact_endpoints(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(frozenset(value)) != len(value) or any(
            endpoint != endpoint.strip() or not endpoint.strip() or "*" in endpoint or "://" in endpoint
            for endpoint in value
        ):
            raise ValueError("Provider endpoints must be unique exact slugs, not URLs or wildcards")
        return value

    @field_validator("allow_fallbacks", mode="before")
    @classmethod
    def no_fallbacks(cls, value: object) -> Literal[False]:
        if value is not False:
            raise ValueError("Offering provider policies require allow_fallbacks=false")
        return False

    def request_body(self) -> Mapping[str, object]:
        return MappingProxyType({"only": list(self.only), "allow_fallbacks": self.allow_fallbacks})

    def allows_endpoint(self, tag: str) -> bool:
        return any(
            tag == slug
            or tag.startswith(slug + "/")
            and not {"fast", "flex"}.intersection(tag[len(slug) + 1 :].split("/"))
            for slug in self.only
        )
