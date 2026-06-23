from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Literal,
    Protocol,
    Union,
)
from urllib.parse import quote

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
DEFAULT_REQUEST_TIMEOUT_SECONDS = 10.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 5.0
# Hard ceiling on the in-memory trust cache so a high-cardinality stream of
# slugs cannot grow it without bound (review: "unbounded in-memory cache").
DEFAULT_MAX_CACHE_ENTRIES = 1024
# Hard ceiling on how many caller-supplied slugs we will look up per request
# when trust_request_slugs is enabled (review: "unbounded slug lookups").
DEFAULT_MAX_REQUEST_SLUGS = 20

# Hlido slugs are lowercase kebab/dot identifiers. We constrain the shape both
# to drop junk early and as defense-in-depth against path injection — only
# values matching this ever reach the request URL (review: "client-controlled
# trust subject" / "URL path injection").
_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_FailureAction = Literal["allow", "block"]


def _is_valid_slug(slug: str) -> bool:
    return bool(_SLUG_RE.match(slug))


class _AsyncGetHandler(Protocol):
    def get(
        self,
        url: str,
        params: dict | None = None,
        headers: dict | None = None,
        follow_redirects: bool | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> Awaitable[httpx.Response]: ...


@dataclass(frozen=True)
class HlidoAgentRecord:
    slug: str
    score: int | None
    tier: str | None
    evidence_url: str | None


@dataclass(frozen=True)
class TrustBlocked:
    slug: str
    reason: str
    evidence_url: str | None


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
        api_base: str | None = None,
        api_key: str | None = None,
        min_score: int | None = None,
        allowed_tiers: tuple[str, ...] | None = None,
        slugs: tuple[str, ...] | None = None,
        on_unverified: str | None = None,
        on_error: str | None = None,
        cache_ttl: float | None = None,
        trust_request_slugs: bool | None = None,
        max_request_slugs: int | None = None,
        request_timeout: float | None = None,
        max_cache_entries: int | None = None,
        async_handler: _AsyncGetHandler | None = None,
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
        # Server-configured slugs are the source of truth. Invalid entries are
        # dropped at init with a warning rather than crashing the proxy.
        self.static_slugs: tuple[str, ...] = self._sanitize_slugs(
            slugs or (), source="config", limit=None
        )
        # SECURITY: caller-supplied metadata.hlido_slugs are IGNORED by default
        # so a request cannot choose its own trust subject (or, when no static
        # slug is set, dodge the check). The proxy admin must opt in explicitly.
        self.trust_request_slugs: bool = bool(trust_request_slugs)
        self.max_request_slugs: int = (
            max_request_slugs
            if max_request_slugs is not None and max_request_slugs > 0
            else DEFAULT_MAX_REQUEST_SLUGS
        )
        self.on_unverified: _FailureAction = (
            "block" if on_unverified == "block" else "allow"
        )
        self.on_error: _FailureAction = "block" if on_error == "block" else "allow"
        self.cache_ttl = (
            cache_ttl if cache_ttl is not None else DEFAULT_CACHE_TTL_SECONDS
        )
        self.max_cache_entries: int = (
            max_cache_entries
            if max_cache_entries is not None and max_cache_entries > 0
            else DEFAULT_MAX_CACHE_ENTRIES
        )
        self._request_timeout: httpx.Timeout = httpx.Timeout(
            (
                request_timeout
                if request_timeout is not None and request_timeout > 0
                else DEFAULT_REQUEST_TIMEOUT_SECONDS
            ),
            connect=DEFAULT_CONNECT_TIMEOUT_SECONDS,
        )
        self._cache: dict[str, tuple[float, HlidoAgentRecord | None]] = {}

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
    def get_config_model() -> type["GuardrailConfigModel"] | None:
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
    ) -> Exception | str | dict | None:
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

    @staticmethod
    def _sanitize_slugs(raw: Any, *, source: str, limit: int | None) -> tuple[str, ...]:
        """Validate, de-dup (preserving order) and (optionally) cap a slug list.

        Invalid slugs are dropped with a warning instead of being trusted —
        this is the single choke point that keeps junk and injection payloads
        out of the request URL.
        """
        if not isinstance(raw, (list, tuple)):
            return ()
        seen: set = set()
        cleaned: list[str] = []
        for value in raw:
            if not isinstance(value, str):
                continue
            slug = value.strip()
            if not slug or slug in seen:
                continue
            if not _is_valid_slug(slug):
                verbose_proxy_logger.warning(
                    "Hlido guardrail: ignoring invalid %s slug %r", source, slug
                )
                continue
            seen.add(slug)
            cleaned.append(slug)
            if limit is not None and len(cleaned) >= limit:
                verbose_proxy_logger.warning(
                    "Hlido guardrail: capping %s slugs at %d", source, limit
                )
                break
        return tuple(cleaned)

    def _collect_slugs(self, data: dict) -> tuple[str, ...]:
        request_slugs: tuple[str, ...] = ()
        if self.trust_request_slugs:
            metadata = data.get("metadata") or data.get("litellm_metadata")
            if isinstance(metadata, dict):
                request_slugs = self._sanitize_slugs(
                    metadata.get("hlido_slugs"),
                    source="request",
                    limit=self.max_request_slugs,
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

    def _cache_get(self, slug: str) -> tuple[float, HlidoAgentRecord | None] | None:
        cached = self._cache.get(slug)
        if cached is None:
            return None
        if cached[0] <= time.monotonic():
            # expired — drop eagerly so it does not count toward the bound
            self._cache.pop(slug, None)
            return None
        return cached

    def _cache_put(
        self, slug: str, expires_at: float, record: HlidoAgentRecord | None
    ) -> None:
        # Refresh-in-place keeps the size stable for repeat slugs.
        if slug not in self._cache and len(self._cache) >= self.max_cache_entries:
            self._evict_one()
        self._cache[slug] = (expires_at, record)

    def _evict_one(self) -> None:
        now = time.monotonic()
        # Prefer evicting an already-expired entry; otherwise the soonest-to-expire.
        expired = [k for k, (exp, _) in self._cache.items() if exp <= now]
        if expired:
            for key in expired:
                self._cache.pop(key, None)
            return
        oldest = min(self._cache.items(), key=lambda kv: kv[1][0])[0]
        self._cache.pop(oldest, None)

    async def _get_agent_record(self, slug: str) -> HlidoAgentRecord | None:
        cached = self._cache_get(slug)
        if cached is not None:
            return cached[1]

        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # slug is validated by _SLUG_RE upstream; quote() is belt-and-braces so a
        # value can never break out of the path segment.
        response = await self.async_handler.get(
            url=f"{self.api_base}/v1/agents/{quote(slug, safe='')}",
            headers=headers,
            timeout=self._request_timeout,
        )
        if response.status_code == 404:
            record: HlidoAgentRecord | None = None
        else:
            response.raise_for_status()
            payload = response.json()
            record = HlidoAgentRecord(
                slug=slug,
                score=payload.get("score"),
                tier=payload.get("tier"),
                evidence_url=payload.get("evidence_url"),
            )
        self._cache_put(slug, time.monotonic() + self.cache_ttl, record)
        return record


__all__ = [
    "HlidoGuardrail",
    "HlidoAgentRecord",
    "TrustAllowed",
    "TrustBlocked",
    "TrustUnverified",
    "TrustLookupFailed",
]
