"""A deployment's cache controls merged with the request's, one control at a time."""

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Final

from litellm.litellm_core_utils.add_retry_fallback_headers import ensure_response_additional_headers
from litellm.litellm_core_utils.core_helpers import without
from litellm.router_rust.invoke.shaping import as_dict
from litellm.types.utils import SupportedCacheControls

ALLOWED_CACHE_CONTROLS_PARAM: Final = "allowed_cache_controls"
IGNORED_CACHE_CONTROLS_HEADER: Final = "x-litellm-ignored-cache-controls"
_CONTROL_NAMES: Final = {"s-max-age": "s-maxage"}
_LOCKABLE_CONTROLS: Final = frozenset({*SupportedCacheControls, "use-cache"})


@dataclasses.dataclass(frozen=True, slots=True)
class ModelCacheControls:
    """``controls`` is None when the attempt keeps the request's and the deployment's ``cache`` as they are."""

    controls: Mapping[str, object] | None
    ignored: tuple[str, ...] = ()


def model_cache_controls(litellm_params: Mapping[str, object], request_cache: object) -> ModelCacheControls:
    """The deployment's ``cache`` is the default for each control and the request overrides the controls
    it sends. When the deployment sets ``allowed_cache_controls``, a request control outside that list is
    ignored and named in ``ignored``."""
    allowed: Final = litellm_params.get(ALLOWED_CACHE_CONTROLS_PARAM)
    deployment_controls: Final = _canonical(as_dict(litellm_params.get("cache")))
    request_controls: Final = as_dict(request_cache)
    if not isinstance(allowed, Sequence) or isinstance(allowed, str):
        if not deployment_controls or not request_controls:
            return ModelCacheControls(None)
        return ModelCacheControls({**deployment_controls, **_canonical(request_controls)})
    ignored: Final = tuple(
        control
        for control in request_controls
        if _CONTROL_NAMES.get(control, control) in _LOCKABLE_CONTROLS
        and _CONTROL_NAMES.get(control, control) not in allowed
    )
    kept: Final = {control: value for control, value in request_controls.items() if control not in ignored}
    return ModelCacheControls({**deployment_controls, **_canonical(kept)}, ignored)


def _canonical(controls: Mapping[str, object]) -> Mapping[str, object]:
    return {_CONTROL_NAMES.get(control, control): value for control, value in controls.items()}


def with_model_cache_controls(
    litellm_params: Mapping[str, object], request: Mapping[str, object]
) -> tuple[Mapping[str, object], Mapping[str, object], tuple[str, ...]]:
    """The deployment's params and the request, each carrying the merged ``cache``, and the ignored controls.
    An empty merge leaves ``cache`` out of both, so the router's default params still apply."""
    cache: Final = model_cache_controls(litellm_params, request.get("cache"))
    params: Final = without(litellm_params, ALLOWED_CACHE_CONTROLS_PARAM)
    if cache.controls is None:
        return params, request, ()
    controls: Final = {"cache": cache.controls} if cache.controls else {}
    return {**without(params, "cache"), **controls}, {**without(request, "cache"), **controls}, cache.ignored


def report_ignored_cache_controls(response: object, ignored: Sequence[str]) -> None:
    if ignored:
        ensure_response_additional_headers(response)[IGNORED_CACHE_CONTROLS_HEADER] = ",".join(ignored)
