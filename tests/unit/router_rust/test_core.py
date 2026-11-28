from collections.abc import Mapping
from typing import Final

import litellm
from litellm.litellm_core_utils.agentic_followup_failure import mark_agentic_followup_failure
from litellm.litellm_core_utils.exception_mapping_utils import keep_routing_classification
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.router_rust.core import attempt_failure

_CLASSIFICATION: Final = ("kind", "status", "status_absent", "retry_after", "detail", "classes", "llm_provider")
_STAMPS: Final = ("num_retries", "failed_deployment_id", "agentic_followup")


def _fields(failure: Mapping[str, object], names: tuple[str, ...]) -> dict[str, object]:
    return {name: failure[name] for name in names}


def _stamped_mapped_error() -> litellm.ServiceUnavailableError:
    mapped: Final = litellm.ServiceUnavailableError(
        message="BedrockException - internalServerException {}", llm_provider="bedrock", model="m", num_retries=2
    )
    mapped.status_code = 500
    mapped.failed_deployment_id = "deployment-1"  # pyright: ignore[reportAttributeAccessIssue]  # the router stamps it
    mark_agentic_followup_failure(mapped, ())
    return mapped


def test_a_mapped_error_is_classified_as_the_error_it_keeps_with_the_stamps_the_router_put_on_it() -> None:
    raw: Final = BedrockError(status_code=500, message="internalServerException {}")
    mapped: Final = _stamped_mapped_error()
    keep_routing_classification(mapped, raw)

    failure: Final = attempt_failure(mapped)

    assert _fields(failure, _CLASSIFICATION) == _fields(attempt_failure(raw), _CLASSIFICATION)
    assert _fields(failure, _STAMPS) == {
        "num_retries": 2,
        "failed_deployment_id": "deployment-1",
        "agentic_followup": True,
    }


def test_a_mapped_error_that_keeps_no_classification_is_classified_as_itself() -> None:
    failure: Final = attempt_failure(_stamped_mapped_error())

    assert failure["kind"] == "ServiceUnavailableError"
    assert failure["status"] == 500
    assert failure["llm_provider"] is True
    assert "ServiceUnavailableError" in str(failure["classes"])
    assert failure["num_retries"] == 2
