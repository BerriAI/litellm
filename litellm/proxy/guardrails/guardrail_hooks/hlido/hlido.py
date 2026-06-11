import time
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Dict,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
)

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.guardrails import GuardrailEventHooks

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import (
        GuardrailConfigModel,
    )

DEFAULT_API_BASE = "https://hlido.eu"
DEFAULT_MIN_SCORE = 60
DEFAULT_CACHE_TTL_SECONDS = 300.0
_REQUEST_TIMEOUT = httpx.Timeout(10.0, connect=5.0)

_FailureAction = Literal["allow", "block"]


class _AsyncGetHandler(Protocol):
    def get(
        self,
        url: str,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        follow_redirects: Optional[bool] = None,
        timeout: Optional[Union[float, httpx.Timeout]] = None,
    ) -> Awaitable[httpx.Response]: ...


@dataclass(frozen=True)
class HlidoAgentRecord:
    slug: str
    score: Optional[int]
    tier: Optional[str]
    evidence_url: Optional[str]


@dataclass(frozen=True)
class TrustBlocked:
    slug: str
    reason: str
    evidence_url: Optional[str]


@dataclass(frozen=True)
class TrustUnverified:
    slug: str


@dataclass(frozen=True)
class TrustLookupFailed:
    slug: str
    error: str


@dataclass(frozen=True)
class TrustAllowed:
    slug: str


_TrustVerdict = Union[TrustAllowed, TrustBlocked, TrustUnverified, TrustLookupFailed]


class HlidoGuardrail(CustomGuardrail):
    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        min_score: Optional[int] = None,
        allowed_tiers: Optional[Tuple[str, ...]] = None,
        slugs: Optional[Tuple[str, ...]] = None,
        on_unverified: Optional[str] = None,
        on_error: Optional[str] = None,
        cache_ttl: Optional[float] = None,
        async_handler: Optional[_AsyncGetHandler] = None,
        **kwargs: Any,
    ) -> None:
        resolved_base = api_base or get_secret_str("HLIDO_API_BASE") or DEFAULT_API_BASE
        self.api_base = resolved_base.rstrip("/")
        self.api_key = api_key or get_secret_str("HLIDO_API_KEY")

        if min_score is None and allowed_tiers is None:
            min_score = DEFAULT_MIN_SCORE
        self.min_score = min_score
        self.allowed_tiers = (
            tuple(tier.upper() for tier in allowed_tiers) if allowed_tiers else None
        )
        self.static_slugs: Tuple[str, ...] = tuple(slugs) if slugs else ()
        self.on_unverified: _FailureAction = (
            "block" if on_unverified == "block" else "allow"
        )
        self.on_error: _FailureAction = "block" if on_error == "block" else "allow"
        self.cache_ttl = (
            cache_ttl if cache_ttl is not None else DEFAULT_CACHE_TTL_SECONDS
        )
        self._cache: Dict[str, Tuple[float, Optional[HlidoAgentRecord]]] = {}

        self.async_handler: _AsyncGetHandler = async_handler or get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback,
        )

        if "supported_event_hooks" not in kwargs:
            kwargs["supported_event_hooks"] = [
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.during_call,
            ]

        super().__init__(**kwargs)

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.hlido import (
            HlidoGuardrailConfigModel,
        )

        return HlidoGuardrailConfigModel

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> Optional[Union[Exception, str, dict]]:
        if (
            self.should_run_guardrail(
                data=data, event_type=GuardrailEventHooks.pre_call
            )
            is not True
        ):
            return data
        await self._check_request(data)
        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        return data

    @log_guardrail_information
    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: str,
    ) -> Any:
        if (
            self.should_run_guardrail(
                data=data, event_type=GuardrailEventHooks.during_call
            )
            is not True
        ):
            return data
        await self._check_request(data)
        return data

    async def _check_request(self, data: dict) -> None:
        slugs = self._collect_slugs(data)
        if not slugs:
            return
        for slug in slugs:
            verdict = await self._evaluate_slug(slug)
            self._raise_on_violation(verdict)

    def _collect_slugs(self, data: dict) -> Tuple[str, ...]:
        request_slugs: Tuple[str, ...] = ()
        metadata = data.get("metadata") or data.get("litellm_metadata")
        if isinstance(metadata, dict):
            raw = metadata.get("hlido_slugs")
            if isinstance(raw, list):
                request_slugs = tuple(
                    slug for slug in raw if isinstance(slug, str) and slug.strip()
                )
        merged = self.static_slugs + tuple(
            slug for slug in request_slugs if slug not in self.static_slugs
        )
        return merged

    async def _evaluate_slug(self, slug: str) -> _TrustVerdict:
        try:
            record = await self._get_agent_record(slug)
        except Exception as exc:
            return TrustLookupFailed(slug=slug, error=str(exc))

        if record is None:
            return TrustUnverified(slug=slug)

        if self.allowed_tiers is not None and (
            record.tier is None or record.tier.upper() not in self.allowed_tiers
        ):
            return TrustBlocked(
                slug=slug,
                reason=(
                    f"tier {record.tier} is not in allowed tiers "
                    f"{list(self.allowed_tiers)}"
                ),
                evidence_url=record.evidence_url,
            )

        if self.min_score is not None and (
            record.score is None or record.score < self.min_score
        ):
            return TrustBlocked(
                slug=slug,
                reason=(
                    f"trust score {record.score} is below the required "
                    f"minimum {self.min_score}"
                ),
                evidence_url=record.evidence_url,
            )

        return TrustAllowed(slug=slug)

    def _raise_on_violation(self, verdict: _TrustVerdict) -> None:
        match verdict:
            case TrustAllowed():
                return
            case TrustBlocked(slug=slug, reason=reason, evidence_url=evidence_url):
                suffix = f" Evidence: {evidence_url}" if evidence_url else ""
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=(
                        f"Hlido trust check failed for agent '{slug}': "
                        f"{reason}.{suffix}"
                    ),
                )
            case TrustUnverified(slug=slug):
                if self.on_unverified == "allow":
                    verbose_proxy_logger.warning(
                        "Hlido guardrail: agent '%s' has no Hlido review; allowing "
                        "per on_unverified=allow",
                        slug,
                    )
                    return
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=(
                        f"Hlido trust check failed for agent '{slug}': no "
                        "independent review exists and on_unverified is 'block'"
                    ),
                )
            case TrustLookupFailed(slug=slug, error=error):
                if self.on_error == "allow":
                    verbose_proxy_logger.warning(
                        "Hlido guardrail: lookup failed for '%s' (%s); allowing "
                        "per on_error=allow",
                        slug,
                        error,
                    )
                    return
                raise GuardrailRaisedException(
                    guardrail_name=self.guardrail_name,
                    message=(
                        f"Hlido trust check could not verify agent '{slug}' "
                        "(API unreachable) and on_error is 'block'"
                    ),
                )

    async def _get_agent_record(self, slug: str) -> Optional[HlidoAgentRecord]:
        cached = self._cache.get(slug)
        if cached is not None and cached[0] > time.monotonic():
            return cached[1]

        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = await self.async_handler.get(
            url=f"{self.api_base}/v1/agents/{slug}",
            headers=headers,
            timeout=_REQUEST_TIMEOUT,
        )
        if response.status_code == 404:
            record: Optional[HlidoAgentRecord] = None
        else:
            response.raise_for_status()
            payload = response.json()
            record = HlidoAgentRecord(
                slug=slug,
                score=payload.get("score"),
                tier=payload.get("tier"),
                evidence_url=payload.get("evidence_url"),
            )
        self._cache[slug] = (time.monotonic() + self.cache_ttl, record)
        return record
