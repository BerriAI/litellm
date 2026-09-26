"""Every credential-bearing param is classified for the credential canary suite.

A new field that can carry a credential needs a canary slot, or the suite's sweeps
never plant a value for it and cannot notice when it is copied somewhere it should
not be. These tests fail until the field is classified below as ``Secret(<slot>)``
or ``NotSecret(<reason>)``.
"""

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from litellm.proxy.auth.auth_utils import is_request_body_safe
from litellm.types.router import LiteLLM_Params, LiteLLMParamsTypedDict
from litellm.types.utils import CustomPricingLiteLLMParams, StandardCallbackDynamicParams

CANARY_SLOTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "A1": "virtual key",
        "A2": "master key",
        "B1": "deployment api_key in config.yaml",
        "B2": "deployment api_key added through the API",
        "B3": "credentials table entry",
        "B4": "cloud provider credentials on a deployment",
        "B5": "team and project model credential overrides",
        "C1": "team callback credentials",
        "C2": "key callback credentials",
        "C3": "team callback credentials for other sinks",
        "D1": "client-side api_key in the request body",
        "D2": "forwarded x-api-key header",
        "D3": "forwarded client headers",
        "D4": "client OAuth token header",
        "D6": "provider credentials sent in the request body",
        "E1": "guardrail api_key",
        "F1": "MCP server static auth",
        "F2": "MCP per-user OAuth token and env vars",
        "F3": "MCP client auth headers",
        "G1": "sink credentials from env",
        "H1": "pass-through endpoint headers",
        "H2": "vector store and search tool credentials",
    }
)

THIS_FILE: Final = "tests/unit/proxy/test_credential_slot_registry.py"

CREDENTIAL_NAME: Final = re.compile(r"(?:^|_)(?:key|secret|token|password|credential)")
"""Matches a name segment that starts with a credential word. Anchoring on a segment start keeps
``valkey_host`` and the other ``valkey_*`` settings out, and still matches ``aws_access_key_id``."""

PRICING_FIELDS: Final = frozenset(CustomPricingLiteLLMParams.model_fields)
"""Excluded from the name match: in ``input_cost_per_token`` and friends, token is a billing unit."""


@dataclass(frozen=True)
class Secret:
    slot: str


@dataclass(frozen=True)
class NotSecret:
    reason: str


Classification = Secret | NotSecret

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
        "gcs_path_service_account": NotSecret("filesystem path to a key file, not the key material"),
        "langsmith_api_key": Secret("C3"),
        "langsmith_project": NotSecret("project name"),
        "langsmith_base_url": NotSecret("sink endpoint URL"),
        "langsmith_sampling_rate": NotSecret("sampling rate"),
        "langsmith_tenant_id": NotSecret("tenant identifier"),
        "humanloop_api_key": Secret("C3"),
        "arize_api_key": Secret("C3"),
        "arize_space_key": Secret("C3"),
        "arize_space_id": NotSecret("space identifier"),
        "arize_success_sampling_rate": NotSecret("sampling rate"),
        "arize_error_sampling_rate": NotSecret("sampling rate"),
        "posthog_api_key": Secret("C3"),
        "posthog_api_url": NotSecret("sink endpoint URL"),
        "wandb_api_key": Secret("C3"),
        "weave_project_id": NotSecret("project identifier"),
        "dd_api_key": Secret("C3"),
        "dd_site": NotSecret("sink site name"),
        "dd_agent_host": NotSecret("agent host name"),
        "dd_agent_port": NotSecret("agent port"),
        "newrelic_api_key": Secret("C3"),
        "newrelic_region": NotSecret("region name"),
        "turn_off_message_logging": NotSecret("boolean logging switch"),
        "litellm_disabled_callbacks": NotSecret("list of callback names"),
    }
)

DEPLOYMENT_PARAM_CLASSIFICATION: Final[Mapping[str, Classification]] = MappingProxyType(
    {
        "api_key": Secret("B1"),
        "azure_ad_token": Secret("B4"),
        "client_secret": Secret("B4"),
        "azure_password": Secret("B4"),
        "vertex_credentials": Secret("B4"),
        "aws_access_key_id": Secret("B4"),
        "aws_secret_access_key": Secret("B4"),
        "aws_session_token": Secret("B4"),
        "aws_web_identity_token": Secret("B4"),
        "s3_access_key_id": Secret("B4"),
        "s3_secret_access_key": Secret("B4"),
        "s3_encryption_key_id": NotSecret("KMS key identifier, not key material"),
        "litellm_credential_name": NotSecret("name of a credentials table entry; its values are slot B3"),
        "default_api_key_tpm_limit": NotSecret("rate limit number"),
        "default_api_key_rpm_limit": NotSecret("rate limit number"),
        "valkey_password": Secret("H2"),
    }
)

REQUEST_BODY_PARAM_CLASSIFICATION: Final[Mapping[str, Classification]] = MappingProxyType(
    {
        "api_key": Secret("D1"),
        "aws_access_key_id": Secret("D6"),
        "aws_secret_access_key": Secret("D6"),
        "aws_session_token": Secret("D6"),
        "azure_password": Secret("D6"),
        "client_secret": Secret("D6"),
        "s3_access_key_id": Secret("D6"),
        "s3_secret_access_key": Secret("D6"),
        "valkey_password": Secret("D6"),
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
        "(pick the slot from CANARY_SLOTS, or add a new slot and a canary for it to the credential canary suite), "
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


def test_every_secret_names_a_known_canary_slot():
    unknown: Final = sorted(
        (mapping_name, param, classification.slot)
        for mapping_name, mapping in (
            ("CALLBACK_PARAM_CLASSIFICATION", CALLBACK_PARAM_CLASSIFICATION),
            ("DEPLOYMENT_PARAM_CLASSIFICATION", DEPLOYMENT_PARAM_CLASSIFICATION),
            ("REQUEST_BODY_PARAM_CLASSIFICATION", REQUEST_BODY_PARAM_CLASSIFICATION),
        )
        for param, classification in mapping.items()
        if isinstance(classification, Secret) and classification.slot not in CANARY_SLOTS
    )
    assert not unknown, f"Secret entries name slots missing from CANARY_SLOTS: {unknown}"
