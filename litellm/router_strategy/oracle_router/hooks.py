"""Post-call hook for the ORACLE router.

After every completion it adds LiteLLM's computed spend and the latest answer to the program's row, and
when the request carried ``metadata.program_done`` it completes the program: the slot is released at once
and the verifier runs in the background. Nothing here can fail a request.
"""

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_router_logger
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
    return as_str_mapping(as_str_mapping(kwargs.get("litellm_params")).get("metadata"))


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
        buckets: Final = (as_str_mapping(data.get("litellm_metadata")), as_str_mapping(data.get("metadata")))
        chosen: Final = next(
            (bucket[CHOSEN_MODEL_METADATA_KEY] for bucket in buckets if CHOSEN_MODEL_METADATA_KEY in bucket), None
        )
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
            program_id: Final = metadata.get(PROGRAM_ID_METADATA_KEY)
            if not isinstance(program_id, str):
                return
            if succeeded:
                cost: Final = kwargs.get("response_cost")
                self.oracle_router.observe_request(
                    program_id,
                    cost=float(cost) if isinstance(cost, (int, float)) else 0.0,
                    response_text=response_text(response_obj),
                    messages=kwargs.get("messages"),
                )
            if metadata.get(PROGRAM_DONE_KEY):
                self.oracle_router.complete(program_id, score=_score(metadata))
        except Exception as error:  # noqa: BLE001  # a logging callback must never fail the request it observes
            verbose_router_logger.exception("OracleRouterPostCallHook: failed to record request: %s", error)
