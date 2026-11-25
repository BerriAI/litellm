from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.constants import INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.sensitive_data_masker import redact_credentials_in_payload
from litellm.types.utils import AUTOROUTER_CLASSIFIER_CALL_ORIGIN, ClassifierAudit

CLASSIFIER_AUDIT_FIELDS: Final = ("classifier_input", "originating_request_masked")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def classifier_input_snapshot(value: object, *, openai_sdk: bool = False) -> Mapping[str, JsonValue] | None:
    try:
        fields: Final = as_str_mapping(value) if openai_sdk else None
        if fields is not None:
            body: Final = MappingProxyType(
                {key: item for key, item in fields.items() if key not in ("extra_headers", "extra_query", "extra_body")}
            )
            extra_body: Final = as_str_mapping(fields.get("extra_body"))
            return _JSON_OBJECT.validate_python(
                MappingProxyType({**body, **extra_body}) if extra_body is not None else body
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
        (metadata := as_str_mapping(params.get(key))) is not None
        and metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY) == AUTOROUTER_CLASSIFIER_CALL_ORIGIN
        for key in ("metadata", "litellm_metadata")
    )


def masked_originating_request(request_kwargs: Mapping[str, object] | None) -> Mapping[str, JsonValue] | None:
    request: Final = as_str_mapping(request_kwargs.get("proxy_server_request")) if request_kwargs is not None else None
    body: Final = as_str_mapping(request.get("body")) if request is not None else None
    if body is None:
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
