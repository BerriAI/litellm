from typing import Final

from pydantic import BaseModel, ConfigDict, Field


class HookFilterConfig(BaseModel):
    """
    Scopes a single hook method on a callback/guardrail to a subset of requests.
    Each field is OR-matched against its own dimension (any pattern matching is
    enough); a filter with more than one dimension set requires all of them to
    match (AND across dimensions). A dimension left unset (``None``) always
    matches; an explicit empty list matches nothing, since it has no patterns
    for any value to satisfy.
    """

    model_config = ConfigDict(frozen=True)

    models: tuple[str, ...] | None = Field(
        default=None,
        description="Glob patterns matched against the requested model group name.",
    )
    key_aliases: tuple[str, ...] | None = Field(
        default=None,
        description="Glob patterns matched against the calling virtual key's key_alias.",
    )
    model_tags: tuple[str, ...] | None = Field(
        default=None,
        description=(
            "Glob patterns matched against the resolved deployment's litellm_params.tags. "
            "Only valid on deployment-scoped hooks, since deployment tags aren't resolved "
            "yet when a pre-routing hook runs."
        ),
    )
    request_tags: tuple[str, ...] | None = Field(
        default=None,
        description="Glob patterns matched against the request's metadata.tags / x-litellm-tags.",
    )


DEPLOYMENT_SCOPED_HOOK_NAMES: Final[frozenset[str]] = frozenset(
    (
        "async_pre_call_deployment_hook",
        "async_post_call_success_deployment_hook",
        "async_post_call_failure_deployment_hook",
        "async_post_call_streaming_deployment_hook",
    )
)

_PROXY_SCOPED_HOOK_NAMES: Final[frozenset[str]] = frozenset(
    (
        "async_pre_call_hook",
        "async_moderation_hook",
        "async_post_call_success_hook",
        "async_post_call_failure_hook",
        "async_post_call_response_headers_hook",
        "async_post_call_streaming_hook",
        "async_post_call_streaming_iterator_hook",
        "logging_hook",
        "async_logging_hook",
        "log_success_event",
        "async_log_success_event",
        "log_failure_event",
        "async_log_failure_event",
    )
)

FILTERABLE_HOOK_NAMES: Final[frozenset[str]] = DEPLOYMENT_SCOPED_HOOK_NAMES | _PROXY_SCOPED_HOOK_NAMES
