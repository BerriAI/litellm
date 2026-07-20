"""Admission control for pass-through endpoints.

Pass-through forwards a client request to an upstream provider using the
proxy's OWN credentials. If the response cannot be priced, the proxy bills the
upstream account and records `spend = 0` against the caller's key. That is
worse than a plain outage: budgets stop binding on that route, and any
reconciliation that splits a provider invoice across keys by gateway-computed
cost will silently redistribute the unpriced key's spend onto keys that priced
correctly.

Cost tracking on pass-through is an allow-list, not a guarantee: a handler must
recognise the provider AND the specific route AND the streaming mode. Routes
outside that allow-list return `$0` with no error.

This module refuses such requests **before the upstream call**. Checking after
the fact is useless — the money is already spent.

Enforcement is opt-in and fail-closed once enabled:

    general_settings:
      passthrough_require_cost_tracking: true
      passthrough_capabilities:
        - provider: anthropic
          methods: [POST]
          path: /v1/messages
          model_source: body            # read `model` from the request body
        - provider: bedrock
          methods: [POST]
          path: /model/{model_id}/converse
          model_source: path:model_id   # read it from the path placeholder

With `passthrough_require_cost_tracking: true` and no matching capability, the
request is rejected. Set it to false (the default) to preserve today's
behaviour.

Scope, stated honestly: this proves a costing path is *registered* and that the
model resolves to an explicit price entry. It cannot prove the upstream will
return usable usage — that is what a live smoke assertion and per-request
zero-cost alerting are for.
"""

import re
from collections.abc import Mapping
from typing import Any

from litellm._logging import verbose_proxy_logger

REQUIRE_COST_TRACKING_SETTING = "passthrough_require_cost_tracking"
CAPABILITIES_SETTING = "passthrough_capabilities"

# `{name}` matches exactly one path segment, so `/model/{id}/converse` cannot
# swallow `/model/a/b/converse`. Keeping placeholders segment-bound is what
# stops a template from widening into the subtree it was meant to exclude.
#
# `{name*}` matches one or MORE segments (greedy, still bounded by the
# template's literal suffix). Needed where the value itself contains slashes:
# Bedrock router aliases (`aws/anthropic/my-alias`) and inference-profile
# paths — with only single-segment placeholders those are inexpressible, so a
# registered capability silently false-denies them.
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)(\*)?\}")


class PassthroughAdmissionError(Exception):
    """Raised when a pass-through request has no registered costing path."""

    def __init__(self, message: str, status_code: int = 403):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def _is_explicitly_true(value: Any) -> bool:
    """True only for a real boolean True or an explicit truthy scalar.

    Deliberately strict. An arbitrary object (a Mock, a config stub, anything
    whose `.get()` returns another object) must NOT count as "enabled" — that
    would turn admission control on by accident and reject every pass-through
    request with a 500.
    """
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    if isinstance(value, int) and not isinstance(value, bool):
        return value == 1
    return False


def _template_to_regex(path_template: str) -> re.Pattern:
    parts: list[str] = []
    last = 0
    for match in _PLACEHOLDER_RE.finditer(path_template):
        parts.append(re.escape(path_template[last : match.start()]))
        segment_pattern = ".+" if match.group(2) else "[^/]+"
        parts.append(f"(?P<{match.group(1)}>{segment_pattern})")
        last = match.end()
    parts.append(re.escape(path_template[last:]))
    # \Z, not $: `$` also matches before a trailing newline, so a path ending
    # in an encoded %0A would satisfy a template it should not.
    return re.compile("^" + "".join(parts) + r"\Z")


def _normalize_path(path: str) -> str:
    """Collapse duplicate slashes and drop a single trailing slash.

    Only cosmetic normalization belongs here. Percent-decoding deliberately does
    NOT: decoding `%2F` into `/` would let a caller synthesise a path that
    matches a narrower template than the one the upstream will actually route.
    """
    if not path:
        return "/"
    collapsed = re.sub(r"/{2,}", "/", path)
    if len(collapsed) > 1 and collapsed.endswith("/"):
        collapsed = collapsed[:-1]
    return collapsed or "/"


def _price_entry_matches_provider(litellm_provider: str, provider: str) -> bool:
    """Is a price-map entry's `litellm_provider` the same billing family as `provider`?

    Deliberately narrow: `azure` must NOT match `azure_ai` — they are different
    upstreams with different prices. Only families litellm itself splits across
    sibling names are aliased.
    """
    if litellm_provider == provider:
        return True
    if provider == "bedrock" and litellm_provider in {"bedrock", "bedrock_converse"}:
        return True
    if provider == "vertex_ai" and litellm_provider.startswith("vertex_ai"):
        return True
    if provider in ("fireworks", "fireworks_ai") and litellm_provider == "fireworks_ai":
        return True
    return False


def _model_is_priced(model: str | None, provider: str | None = None) -> bool:
    """True when `model` resolves to an explicit entry in the price map.

    A fallback or default price is treated as unpriced on purpose: a wrong
    non-zero cost is harder to detect than a zero one, because nothing looks
    broken.

    When `provider` is given, the entry must belong to THAT provider's billing
    family. A price existing *anywhere* is not enough: admitting `gpt-4` onto
    an Azure route because `openai/gpt-4` is priced still records $0, because
    the success handler prices under the azure key. Ambiguity fails closed —
    a false deny is visible and fixable; a $0 row is silent.
    """
    if not model:
        return False

    import litellm

    def _bare_key_ok(key: str) -> bool:
        if provider is None:
            return True
        info = litellm.model_cost.get(key) or {}
        return _price_entry_matches_provider(str(info.get("litellm_provider") or ""), provider)

    if model in litellm.model_cost and _bare_key_ok(model):
        return True
    # The provider-prefixed key form (`azure/gpt-4`): the prefix itself names
    # the billing family, so it is authoritative without an alias check.
    if provider is not None and f"{provider}/{model}" in litellm.model_cost:
        return True
    # Providers commonly return a bare id where the map is prefixed
    # (`fireworks_ai/accounts/...`), or the reverse. Accept a prefixed form only
    # if it exists explicitly — and, when scoped, only under this provider.
    if "/" in model:
        prefix, bare = model.split("/", 1)
        if bare in litellm.model_cost and (provider is None or prefix == provider or _bare_key_ok(bare)):
            return True
    return any(
        key.endswith("/" + model)
        and (provider is None or _price_entry_matches_provider(key.split("/", 1)[0], provider))
        for key in litellm.model_cost  # type: ignore[union-attr]
    )


def _router_model_is_priced(model: str) -> bool:
    """True when `model` resolves to a router deployment that can price.

    Router-based pass-through (Bedrock) carries a router alias in the URL, not
    a raw price-map key. The costing path for those requests IS the router, so
    admission must accept an alias whose deployment prices — via an explicit
    per-token cost on the deployment, or a priced underlying/base model.
    """
    try:
        from litellm.proxy.proxy_server import llm_router
    except Exception:  # noqa: BLE001  # no proxy server (unit tests, SDK use) just means "no router pricing"
        return False
    if llm_router is None:
        return False
    try:
        deployments = llm_router.get_model_list(model_name=model) or []
    except Exception:  # noqa: BLE001  # a router lookup error must deny (fail closed), not 500 the request
        return False
    for deployment in deployments:
        litellm_params = deployment.get("litellm_params") or {}
        if litellm_params.get("input_cost_per_token") is not None:
            return True
        model_info = deployment.get("model_info") or {}
        for candidate in (model_info.get("base_model"), litellm_params.get("model")):
            if candidate and _model_is_priced(str(candidate)):
                return True
    return False


def _extract_model(
    capability: dict[str, Any],
    path_match: re.Match | None,
    request_body: dict | None,
) -> str | None:
    source = str(capability.get("model_source") or "body")
    if source.startswith("path:"):
        placeholder = source.split(":", 1)[1]
        if path_match is None:
            return None
        try:
            return path_match.group(placeholder)
        except (IndexError, KeyError):
            # Capability declares a placeholder its own path template does not
            # define — a config error, not a client error. Treat as unpriced so
            # it fails closed rather than forwarding unmetered.
            return None
    if not isinstance(request_body, dict):
        return None
    model = request_body.get("model")
    return str(model) if model else None


def find_matching_capability(
    capabilities: list[dict[str, Any]],
    provider: str | None,
    method: str,
    path: str,
) -> tuple[dict[str, Any] | None, re.Match | None]:
    normalized = _normalize_path(path)
    upper_method = (method or "").upper()
    for capability in capabilities:
        if not isinstance(capability, dict):
            continue
        cap_provider = capability.get("provider")
        # A provider-scoped capability requires a KNOWN matching provider.
        # Entry points that pass provider=None (e.g. the vertex router) only
        # match capabilities that don't declare one — otherwise the provider
        # constraint would silently not bind exactly where identity is weakest.
        if cap_provider and (provider is None or str(cap_provider) != str(provider)):
            continue
        methods = capability.get("methods") or []
        if methods and upper_method not in {str(m).upper() for m in methods}:
            continue
        template = str(capability.get("path") or "")
        if not template:
            continue
        try:
            pattern = _template_to_regex(_normalize_path(template))
        except re.error as exc:
            # e.g. `/x/{id}/y/{id}` — duplicate group names. A config error,
            # not a client error; name it instead of surfacing an opaque 500.
            raise PassthroughAdmissionError(
                f"Invalid capability path template {template!r}: {exc}",
                status_code=500,
            )
        match = pattern.match(normalized)
        if match:
            return capability, match
    return None, None


def enforce_passthrough_admission(
    general_settings: dict | None,
    provider: str | None,
    method: str,
    path: str,
    request_body: dict | None,
) -> None:
    """Raise PassthroughAdmissionError if this request has no costing path.

    No-op unless `passthrough_require_cost_tracking` is true.
    """
    # Enforcement must be turned on EXPLICITLY. Never infer it from
    # truthiness: `general_settings` is not guaranteed to be a plain dict, and
    # an object whose `.get()` returns another object would otherwise switch
    # enforcement on by accident and reject every pass-through request.
    if not isinstance(general_settings, Mapping):
        return
    if not _is_explicitly_true(general_settings.get(REQUIRE_COST_TRACKING_SETTING, False)):
        return

    capabilities = general_settings.get(CAPABILITIES_SETTING) or []
    if not isinstance(capabilities, list):
        raise PassthroughAdmissionError(f"{CAPABILITIES_SETTING} must be a list of capability objects", status_code=500)

    capability, path_match = find_matching_capability(capabilities, provider, method, path)
    if capability is None:
        verbose_proxy_logger.warning(
            "pass-through admission denied: no registered capability for provider=%s %s %s",
            provider,
            method,
            path,
        )
        raise PassthroughAdmissionError(
            f"Pass-through endpoint '{method} {path}' is not a registered capability"
            f"{f' for provider {provider}' if provider else ''}. "
            "It would bill the upstream provider without recording cost against your key. "
            f"Register it under general_settings.{CAPABILITIES_SETTING} once its cost tracking is verified."
        )

    # `require_priced_model` defaults to True: a capability is registered
    # precisely because we intend to price it.
    if capability.get("require_priced_model", True):
        model = _extract_model(capability, path_match, request_body)
        cap_provider = capability.get("provider") or provider
        priced = _model_is_priced(model, provider=str(cap_provider) if cap_provider else None) or (
            model is not None and _router_model_is_priced(model)
        )
        if not priced:
            verbose_proxy_logger.warning(
                "pass-through admission denied: model %r has no explicit price entry (%s %s)",
                model,
                method,
                path,
            )
            raise PassthroughAdmissionError(
                f"Model '{model}' has no explicit price entry, so this pass-through request "
                "would be recorded at $0 while still billing the upstream provider. "
                "Add the model to the price map, or set require_priced_model: false on the "
                "capability if it is genuinely non-billable."
            )
