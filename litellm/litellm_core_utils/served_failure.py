"""A response the provider served and billed, but that the caller must see as an error.

The provider transformation marks the response. Logging, billing and router health treat it as the
success it is, and the response edge (``@client`` for a whole response, ``CustomStreamWrapper`` at the
end of a stream) raises ``ServedResponseFailedError`` only after that success was logged or parked."""

from typing import Final

from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.exceptions import InternalServerError, MidStreamFallbackError
from litellm.litellm_core_utils.hidden_params import get_hidden_params, set_hidden_param

SERVED_FAILURE_HIDDEN_PARAM: Final = "served_failure"


class ServedFailure(BaseModel):
    model_config = ConfigDict(frozen=True)

    reason: str
    message: str


class ServedResponseFailedError(InternalServerError):
    def __init__(self, failure: ServedFailure, served_response: object, model: str, llm_provider: str) -> None:
        super().__init__(message=failure.message, llm_provider=llm_provider, model=model)
        self.reason: Final = failure.reason
        self.served_response: Final = served_response


def mark_served_failure(response: object, failure: ServedFailure) -> None:
    set_hidden_param(response, SERVED_FAILURE_HIDDEN_PARAM, failure.model_dump())


def served_failure_of(response: object) -> ServedFailure | None:
    hidden_params: Final = get_hidden_params(response)
    marker: Final = None if hidden_params is None else hidden_params.get(SERVED_FAILURE_HIDDEN_PARAM)
    if marker is None:
        return None
    try:
        return ServedFailure.model_validate(marker)
    except ValidationError:
        return None


def served_failure_error(response: object, model: str) -> ServedResponseFailedError | None:
    failure: Final = served_failure_of(response)
    if failure is None:
        return None
    hidden_params: Final = get_hidden_params(response) or {}
    provider: Final = hidden_params.get("custom_llm_provider")
    return ServedResponseFailedError(failure, response, model, provider if isinstance(provider, str) else "")


def served_failure_in(error: BaseException | None) -> ServedResponseFailedError | None:
    provider_error: Final = error.original_exception if isinstance(error, MidStreamFallbackError) else error
    return provider_error if isinstance(provider_error, ServedResponseFailedError) else None


def is_served_failure(error: BaseException | None) -> bool:
    return served_failure_in(error) is not None
