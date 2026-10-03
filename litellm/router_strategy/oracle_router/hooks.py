"""Post-call hook for the ORACLE router.

After every completion it adds LiteLLM's computed spend and the latest answer to the program's row, and
when the request carried ``metadata.program_done`` it completes the program: the slot is released at once
and the verifier runs in the background. Only requests the Router stamped as routed by this ORACLE router
are read, and only against the program of the key that sent them. Nothing here can fail a request, and
nothing here logs: a failure is kept on the router for ``/oracle_router/state``.
"""

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

from litellm.integrations.custom_logger import CustomLogger
from litellm.router_strategy.oracle_router.config import (
    CHOSEN_MODEL_METADATA_KEY,
    PROGRAM_DONE_KEY,
    PROGRAM_ID_METADATA_KEY,
    PROGRAM_SCORE_KEY,
    RESPONSE_HEADER,
)
from litellm.router_strategy.oracle_router.oracle_router import OracleRouter, as_str_mapping
from litellm.router_strategy.oracle_router.verifier import response_text

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth


def _request_metadata(kwargs: Mapping[str, object]) -> Mapping[str, object]:
    """The request's metadata as the proxy saw it: the caller's bucket with the proxy-internal one on top."""
    litellm_params: Final = as_str_mapping(kwargs.get("litellm_params"))
    return _merged(litellm_params.get("metadata"), litellm_params.get("litellm_metadata"))


def _merged(metadata: object, litellm_metadata: object) -> Mapping[str, object]:
    return {**as_str_mapping(metadata), **as_str_mapping(litellm_metadata)}


def _routed_by(metadata: Mapping[str, object], router_name: str) -> bool:
    """True when the Router stamped this request as routed by this ORACLE router.

    The Router writes or clears ``routing_decision`` on every routing attempt, so a caller cannot make a
    request to some other model look like one of this router's by putting ORACLE keys in its metadata.
    """
    decision: Final = as_str_mapping(metadata.get("routing_decision"))
    return decision.get("router_type") == "oracle" and decision.get("router_model_name") == router_name


def _owner(metadata: Mapping[str, object]) -> str | None:
    """The key the proxy authenticated, stamped server-side as ``user_api_key_hash``; None outside the proxy."""
    value: Final = metadata.get("user_api_key_hash")
    return str(value) if value else None


def _score(metadata: Mapping[str, object]) -> float | None:
    raw: Final = metadata.get(PROGRAM_SCORE_KEY)
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return None
    try:
        score: Final = float(raw)
    except ValueError:
        return None
    return score if math.isfinite(score) else None  # a NaN would corrupt a bandit cell for good


class OracleRouterPostCallHook(CustomLogger):
    """One hook per OracleRouter, registered in ``litellm.callbacks``."""

    def __init__(self, oracle_router: OracleRouter) -> None:
        self.oracle_router: Final[OracleRouter] = oracle_router

    async def async_post_call_response_headers_hook(
        self,
        data: Mapping[str, object],
        user_api_key_dict: "UserAPIKeyAuth",
        response: object,
        request_headers: dict[str, str] | None = None,  # mutable-ok: CustomLogger's hook signature
        litellm_call_info: dict[str, object] | None = None,  # mutable-ok: CustomLogger's hook signature
    ) -> dict[str, str] | None:  # mutable-ok: CustomLogger's hook signature
        metadata: Final = _merged(data.get("metadata"), data.get("litellm_metadata"))
        if not _routed_by(metadata, self.oracle_router.router_name):
            return None
        chosen: Final = metadata.get(CHOSEN_MODEL_METADATA_KEY)
        return {RESPONSE_HEADER: str(chosen)} if chosen else None

    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._record(kwargs, response_obj, succeeded=True)

    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self._record(kwargs, response_obj, succeeded=False)

    def _record(self, kwargs: Mapping[str, object], response_obj: object, succeeded: bool) -> None:
        try:
            metadata: Final = _request_metadata(kwargs)
            if not _routed_by(metadata, self.oracle_router.router_name):
                return
            program_id: Final = metadata.get(PROGRAM_ID_METADATA_KEY)
            if not isinstance(program_id, str):
                return
            owner: Final = _owner(metadata)
            if succeeded:
                cost: Final = kwargs.get("response_cost")
                self.oracle_router.observe_request(
                    program_id,
                    cost=float(cost) if isinstance(cost, (int, float)) else 0.0,
                    response_text=response_text(response_obj),
                    messages=kwargs.get("messages"),
                    owner=owner,
                )
            if metadata.get(PROGRAM_DONE_KEY):
                self.oracle_router.complete(program_id, score=_score(metadata), owner=owner)
        except Exception as error:  # noqa: BLE001  # a logging callback must never fail the request it observes
            self.oracle_router.note_failure("post-call hook", error)
