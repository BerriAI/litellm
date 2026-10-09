"""Every credential-bearing param is classified for the credential canary suite.

These tests fail until a param is classified below as one of:

- ``Secret(<slot id>)``: an integration test in ``tests/integration/security`` plants a canary in
  exactly this param under that slot id.
- ``Unplanted()``: the param can carry a credential, but no integration test plants a canary
  in it yet. This is a classification only.
- ``NotSecret(<reason>)``: the param cannot carry a credential.

``CANARY_SLOTS`` mirrors ``SLOTS`` in ``tests/integration/security/_canary.py``, limited to the ids
whose test plants a canary in one of these params.
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from litellm.proxy.auth.auth_utils import is_request_body_safe
from litellm.types.router import LiteLLM_Params, LiteLLMParamsTypedDict
from litellm.types.utils import CustomPricingLiteLLMParams, StandardCallbackDynamicParams

CANARY_SLOTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "B1": "deployment api_key in config.yaml",
        "B4": "deployment aws_secret_access_key added through /model/new",
        "B4v": "deployment vertex_credentials added through /model/new",
        "C1": "team callback langfuse_secret / langfuse_secret_key",
        "C3": "team callback dd_api_key for the Datadog sink",
        "D1": "client-side api_key in the request body",
    }
)

THIS_FILE: Final = "tests/unit/proxy/test_credential_slot_registry.py"

HARNESS_FILE: Final = Path(__file__).resolve().parents[2] / "integration" / "security" / "_canary.py"

CREDENTIAL_NAME: Final = re.compile(r"(?:^|_)(?:key|secret|token|password|credential)")
"""Matches a name segment that starts with a credential word. Anchoring on a segment start keeps
``valkey_host`` and the other ``valkey_*`` settings out, and still matches ``aws_access_key_id``."""

PRICING_FIELDS: Final = frozenset(CustomPricingLiteLLMParams.model_fields)
"""Excluded from the name match: in ``input_cost_per_token`` and friends, token is a billing unit."""


@dataclass(frozen=True)
class Secret:
    slot: str

    def __post_init__(self) -> None:
        if self.slot not in CANARY_SLOTS:
            raise ValueError(f"Secret({self.slot!r}) names no slot in CANARY_SLOTS")


@dataclass(frozen=True)
class Unplanted:
    pass


@dataclass(frozen=True)
class NotSecret:
    reason: str


Classification = Secret | Unplanted | NotSecret

CALLBACK_PARAM_CLASSIFICATION: Final[Mapping[str, Classification]] = MappingProxyType(
    {
        "langfuse_public_key": NotSecret("public half of the Langfuse key pair, an identifier"),
        "langfuse_secret": Secret("C1"),
        "langfuse_secret_key": Secret("C1"),
        "langfuse_host": NotSecret("sink endpoint URL"),
        "langfuse_environment": NotSecret("environment label"),
        "langfuse_span_scope": NotSecret("span scope setting"),
        "langfuse_prompt_version": NotSecret("prompt version number"),
        "gcs_bucket_name": NotSecret("bucket name"),
        "gcs_path_service_account": Unplanted(),
        "langsmith_api_key": Unplanted(),
        "langsmith_project": NotSecret("project name"),
        "langsmith_base_url": NotSecret("sink endpoint URL"),
        "langsmith_sampling_rate": NotSecret("sampling rate"),
        "langsmith_tenant_id": NotSecret("tenant identifier"),
        "humanloop_api_key": Unplanted(),
        "arize_api_key": Unplanted(),
        "arize_space_key": Unplanted(),
        "arize_space_id": NotSecret("space identifier"),
        "arize_success_sampling_rate": NotSecret("sampling rate"),
        "arize_error_sampling_rate": NotSecret("sampling rate"),
        "arize_otlp_protocol": NotSecret("transport protocol enum"),
        "posthog_api_key": Unplanted(),
        "posthog_api_url": NotSecret("sink endpoint URL"),
        "wandb_api_key": Unplanted(),
        "weave_project_id": NotSecret("project identifier"),
        "dd_api_key": Secret("C3"),
        "dd_site": NotSecret("sink site name"),
        "dd_agent_host": NotSecret("agent host name"),
        "dd_agent_port": NotSecret("agent port"),
        "newrelic_api_key": Unplanted(),
        "newrelic_region": NotSecret("region name"),
        "signoz_ingestion_key": Unplanted(),
        "signoz_ingestion_endpoint": NotSecret("sink endpoint URL"),
        "turn_off_message_logging": NotSecret("boolean logging switch"),
        "litellm_disabled_callbacks": NotSecret("list of callback names"),
    }
)

DEPLOYMENT_PARAM_CLASSIFICATION: Final[Mapping[str, Classification]] = MappingProxyType(
    {
        "api_key": Secret("B1"),
        "azure_ad_token": Unplanted(),
        "client_secret": Unplanted(),
        "azure_password": Unplanted(),
        "vertex_credentials": Secret("B4v"),
        "aws_access_key_id": Unplanted(),
        "aws_secret_access_key": Secret("B4"),
        "aws_session_token": Unplanted(),
        "aws_web_identity_token": Unplanted(),
        "s3_access_key_id": Unplanted(),
        "s3_secret_access_key": Unplanted(),
        "s3_encryption_key_id": NotSecret("KMS key identifier, not key material"),
        "litellm_credential_name": NotSecret("name of a credentials table entry, not a credential"),
        "default_api_key_tpm_limit": NotSecret("rate limit number"),
        "default_api_key_rpm_limit": NotSecret("rate limit number"),
        "valkey_password": Unplanted(),
        "anthropic_identity_token": Unplanted(),
        "anthropic_identity_token_file": Unplanted(),
        "anthropic_issuer_signing_key_ref": Unplanted(),
        "anthropic_keycloak_token_url": NotSecret("Keycloak token endpoint URL"),
        "anthropic_keycloak_client_id": NotSecret("Keycloak client identifier"),
        "anthropic_keycloak_auth_method": NotSecret("name of the client authentication method"),
        "anthropic_keycloak_client_secret_ref": Unplanted(),
        "anthropic_keycloak_scope": NotSecret("OAuth scope string"),
        "openai_identity_token_file": Unplanted(),
    }
)

REQUEST_BODY_PARAM_CLASSIFICATION: Final[Mapping[str, Classification]] = MappingProxyType(
    {
        "api_key": Secret("D1"),
        "aws_access_key_id": Unplanted(),
        "aws_secret_access_key": Unplanted(),
        "aws_session_token": Unplanted(),
        "azure_password": Unplanted(),
        "client_secret": Unplanted(),
        "s3_access_key_id": Unplanted(),
        "s3_secret_access_key": Unplanted(),
        "valkey_password": Unplanted(),
        "s3_encryption_key_id": NotSecret("KMS key identifier, not key material"),
        "litellm_credential_name": NotSecret("name of a credentials table entry, not a credential"),
        "default_api_key_tpm_limit": NotSecret("rate limit number"),
        "default_api_key_rpm_limit": NotSecret("rate limit number"),
    }
)


def _credential_named(names: Iterable[str]) -> frozenset[str]:
    return frozenset(name for name in names if CREDENTIAL_NAME.search(name)) - PRICING_FIELDS


def _deployment_param_names() -> frozenset[str]:
    return (
        frozenset(LiteLLM_Params.model_fields)
        | LiteLLMParamsTypedDict.__required_keys__
        | LiteLLMParamsTypedDict.__optional_keys__
    )


def _callback_param_names() -> frozenset[str]:
    return StandardCallbackDynamicParams.__required_keys__ | StandardCallbackDynamicParams.__optional_keys__


def _accepted_in_request_body(param: str) -> bool:
    try:
        return is_request_body_safe({"model": "m", param: "v"}, general_settings={}, llm_router=None, model="m")
    except ValueError:
        return False


def _assert_classified(
    source: str, names: frozenset[str], mapping: Mapping[str, Classification], mapping_name: str
) -> None:
    unclassified: Final = sorted(names - mapping.keys())
    stale: Final = sorted(mapping.keys() - names)
    assert not unclassified, (
        f"{source} has params with no credential classification: {unclassified}. "
        f"Add each to {mapping_name} in {THIS_FILE} as Secret('<slot id>') if it can hold a credential "
        "and an integration test plants it under a slot in CANARY_SLOTS, as Unplanted() if it can hold a credential "
        "but no integration test plants it yet, "
        "or as NotSecret('<one-line reason>') if it cannot."
    )
    assert not stale, f"{mapping_name} in {THIS_FILE} classifies params {source} no longer has: {stale}. Remove them."


def test_every_callback_dynamic_param_is_classified():
    _assert_classified(
        "StandardCallbackDynamicParams",
        _callback_param_names(),
        CALLBACK_PARAM_CLASSIFICATION,
        "CALLBACK_PARAM_CLASSIFICATION",
    )


def test_every_credential_named_deployment_param_is_classified():
    _assert_classified(
        "LiteLLM_Params / LiteLLMParamsTypedDict",
        _credential_named(_deployment_param_names()),
        DEPLOYMENT_PARAM_CLASSIFICATION,
        "DEPLOYMENT_PARAM_CLASSIFICATION",
    )


def test_every_credential_named_param_a_client_may_send_is_classified():
    candidates: Final = _credential_named(_deployment_param_names() | _callback_param_names())
    _assert_classified(
        "is_request_body_safe with default settings",
        frozenset(name for name in candidates if _accepted_in_request_body(name)),
        REQUEST_BODY_PARAM_CLASSIFICATION,
        "REQUEST_BODY_PARAM_CLASSIFICATION",
    )


def test_every_canary_slot_exists_in_the_harness():
    harness_slots: Final = frozenset(re.findall(r'^\s+"(\w+)": Slot\(', HARNESS_FILE.read_text(), re.MULTILINE))
    assert harness_slots, f"found no Slot(...) entries in {HARNESS_FILE}"
    missing: Final = sorted(CANARY_SLOTS.keys() - harness_slots)
    assert not missing, f"CANARY_SLOTS in {THIS_FILE} names slots {HARNESS_FILE.name} does not define: {missing}"
