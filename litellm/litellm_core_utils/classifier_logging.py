from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.sensitive_data_masker import redact_credentials_in_payload
from litellm.types.utils import AUTOROUTER_CLASSIFIER_CALL_ORIGIN, ClassifierAudit

CLASSIFIER_AUDIT_FIELDS: Final = ("classifier_input", "originating_request_masked")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def classifier_input_snapshot(value: object, *, openai_sdk: bool = False) -> Mapping[str, JsonValue] | None:
    try:
        if openai_sdk and isinstance(value, Mapping):
            body: Final = MappingProxyType(
                {key: item for key, item in value.items() if key not in ("extra_headers", "extra_query", "extra_body")}
            )
            extra_body: Final = value.get("extra_body")
            return _JSON_OBJECT.validate_python(
                MappingProxyType({**body, **extra_body}) if isinstance(extra_body, Mapping) else body
            )
        return (
            _JSON_OBJECT.validate_json(value)
            if isinstance(value, (str, bytes))
            else _JSON_OBJECT.validate_python(value)
        )
    except ValidationError:
        return None


def is_classifier_call(call_type: str, params: Mapping[str, object]) -> bool:
    return call_type in ("completion", "acompletion", "responses", "aresponses") and any(
        isinstance(metadata := params.get(key), Mapping)
        and metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY) == AUTOROUTER_CLASSIFIER_CALL_ORIGIN
        for key in ("metadata", "litellm_metadata")
    )


def masked_originating_request(request_kwargs: Mapping[str, object] | None) -> Mapping[str, JsonValue] | None:
    request: Final = request_kwargs.get("proxy_server_request") if request_kwargs is not None else None
    body: Final = request.get("body") if isinstance(request, Mapping) else None
    if not isinstance(body, Mapping):
        return None
    serializable: Final = classifier_input_snapshot(safe_dumps(body))
    return classifier_input_snapshot(redact_credentials_in_payload(serializable)) if serializable is not None else None


def classifier_audit_fields(payload: Mapping[str, object]) -> ClassifierAudit:
    classifier_input: Final = classifier_input_snapshot(payload.get("classifier_input"))
    originating_request: Final = classifier_input_snapshot(payload.get("originating_request_masked"))
    if classifier_input is None:
        return (
            ClassifierAudit(originating_request_masked=originating_request)
            if originating_request is not None
            else ClassifierAudit()
        )
    if originating_request is None:
        return ClassifierAudit(classifier_input=classifier_input)
    return ClassifierAudit(classifier_input=classifier_input, originating_request_masked=originating_request)


def without_classifier_audit(payload: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if key not in CLASSIFIER_AUDIT_FIELDS}
