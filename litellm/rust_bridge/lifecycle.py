from __future__ import annotations

import datetime
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Protocol, cast  # noqa: TID251  # bounded compatibility calls into legacy Python integrations

import litellm
from litellm import utils
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.llm_response_utils import response_metadata


class CredentialLoader(Protocol):
    def __call__(self, kwargs: dict[str, object]) -> None: ...


class MetadataUpdater(Protocol):
    def __call__(
        self,
        result: object,
        logging_obj: Logging,
        model: str | None,
        kwargs: dict[str, object],
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class CallSetup:
    logger: Logging
    kwargs: dict[str, object]


def setup(call_type: str, kwargs: Mapping[str, object], start_time: datetime.datetime, asynchronous: bool) -> CallSetup:
    arguments: Final = {  # mutable-ok: function_setup consumes an owned kwargs dict
        "litellm_call_id": str(uuid.uuid4()),
        **kwargs,
    }
    supplied: Final = arguments.get("litellm_logging_obj")
    if isinstance(supplied, Logging):
        return CallSetup(supplied, arguments)
    logger, prepared = utils.function_setup(
        call_type, utils.Rules(), start_time, is_async_call=asynchronous, **arguments
    )
    return CallSetup(logger, prepared)


def prepare(kwargs: Mapping[str, object], logger: Logging) -> dict[str, object]:
    arguments: Final = {  # mutable-ok: credential loader updates an owned kwargs dict
        **kwargs,
        "litellm_logging_obj": logger,
    }
    load_credentials: Final = cast(  # cast-ok: legacy credential loader mutates a concrete kwargs dict
        CredentialLoader, utils.load_credentials_from_list
    )
    load_credentials(arguments)
    current_cost: Final = litellm._current_cost  # pyright: ignore[reportPrivateUsage]  # shared SDK budget counter has no public accessor
    if litellm.max_budget and current_cost > litellm.max_budget:
        raise litellm.BudgetExceededError(current_cost=current_cost, max_budget=litellm.max_budget)
    metadata: Final = arguments.get("metadata")
    if isinstance(metadata, Mapping):
        typed_metadata: Final = cast(  # cast-ok: runtime Mapping check establishes read-only metadata
            Mapping[str, object], metadata
        )
        previous: Final = typed_metadata.get("previous_models")
        if (
            isinstance(previous, list)
            and litellm.num_retries_per_request is not None
            and len(cast(list[object], previous))  # cast-ok: runtime list check establishes the retry history
            >= litellm.num_retries_per_request
        ):
            raise RuntimeError("Max retries per request hit!")
    return arguments


def finalize(
    response: object,
    logger: Logging,
    kwargs: dict[str, object],
    start_time: datetime.datetime,
    end_time: datetime.datetime,
) -> None:
    model: Final = kwargs.get("model")
    update: Final = cast(  # cast-ok: legacy metadata function accepts concrete kwargs
        MetadataUpdater, response_metadata.update_response_metadata
    )
    update(response, logger, model if isinstance(model, str) else None, kwargs, start_time, end_time)
