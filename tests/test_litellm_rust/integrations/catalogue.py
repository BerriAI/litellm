from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

DependencyProfile = Literal["required", "enterprise"]

OSS_LOGGER_NAMES: Final = frozenset(
    {
        "agentops",
        "anthropic_cache_control_hook",
        "argilla",
        "arize",
        "arize_phoenix",
        "aws_sqs",
        "azure_sentinel",
        "azure_storage",
        "bitbucket",
        "braintrust",
        "cloudzero",
        "datadog",
        "datadog_llm_observability",
        "datadog_metrics",
        "deepeval",
        "dotprompt",
        "dynamic_rate_limiter",
        "dynamic_rate_limiter_v3",
        "focus",
        "galileo",
        "gcs_bucket",
        "gcs_pubsub",
        "gitlab",
        "humanloop",
        "lago",
        "langfuse",
        "langfuse_otel",
        "langsmith",
        "langtrace",
        "levo",
        "literalai",
        "litellm_agent",
        "logfire",
        "mavvrik",
        "mlflow",
        "newrelic",
        "opentelemetry",
        "openmeter",
        "opik",
        "otel",
        "posthog",
        "prometheus",
        "s3_v2",
        "vantage",
        "vector_store_pre_call_hook",
        "weave_otel",
    }
)
ENTERPRISE_LOGGER_NAMES: Final = frozenset({"generic_api", "pagerduty", "resend_email", "sendgrid_email", "smtp_email"})
GUARDRAIL_NAMES: Final = frozenset(
    {
        "aim",
        "akto",
        "alice",
        "aporia",
        "azure/prompt_shield",
        "azure/text_moderations",
        "bedrock",
        "block_code_execution",
        "cato_networks",
        "cisco_ai_defense",
        "compresr",
        "crowdstrike_aidr",
        "custom_code",
        "deepkeep",
        "dynamoai",
        "enkryptai",
        "generic_guardrail_api",
        "grayswan",
        "guardrails_ai",
        "headroom",
        "hiddenlayer",
        "hide-secrets",
        "ibm_guardrails",
        "javelin",
        "lakera",
        "lakera_v2",
        "lasso",
        "litellm_content_filter",
        "llm_as_a_judge",
        "mcp_end_user_permission",
        "mcp_jwt_signer",
        "mcp_security",
        "microsoft_purview",
        "model_armor",
        "noma",
        "noma_v2",
        "onyx",
        "openai_moderation",
        "ovalix",
        "pangea",
        "panw_prisma_airs",
        "pillar",
        "presidio",
        "prompt_security",
        "promptguard",
        "qostodian_nexus",
        "qualifire",
        "repelloai",
        "rubrik",
        "semantic_guard",
        "singulr",
        "straiker",
        "tool_permission",
        "vigil_guard",
        "xecguard",
        "zscaler_ai_guard",
    }
)
DISCOVERED_ONLY_GUARDRAIL_NAMES: Final = frozenset({"tool_policy"})


@dataclass(frozen=True, slots=True)
class CoverageObligation:
    stable_name: str
    registration_names: tuple[str, ...]
    dependency_profile: DependencyProfile
    behavioral_cases: tuple[str, ...]


LOGGER_BEHAVIORAL_CASES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "generic_api": ("generic-api-success",),
        "gcs_bucket": ("gcs-literalai-scheduling",),
        "literalai": ("gcs-literalai-scheduling",),
        "prometheus": ("prometheus-string-registration",),
        "opentelemetry": ("otel-export",),
    }
)
GUARDRAIL_BEHAVIORAL_CASES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "azure/text_moderations": ("azure-text-moderation",),
        "crowdstrike_aidr": ("crowdstrike-redaction-native-chat",),
        "rubrik": ("rubrik-block-native-chat",),
        "microsoft_purview": ("purview-audit-native-chat",),
        "litellm_content_filter": ("content-filter-block",),
    }
)


def _logger_obligations() -> Mapping[str, CoverageObligation]:
    from litellm.litellm_core_utils.custom_logger_registry import CustomLoggerRegistry

    registry: Final = CustomLoggerRegistry.CALLBACK_CLASS_STR_TO_CLASS_TYPE
    expected_names: Final = OSS_LOGGER_NAMES | ENTERPRISE_LOGGER_NAMES
    grouped: Final[dict[type[object], tuple[str, ...]]] = {
        implementation: tuple(sorted(name for name in expected_names if registry.get(name) is implementation))
        for implementation in frozenset(registry.values())
    }
    obligations: Final = {
        names[0]: CoverageObligation(
            stable_name=names[0],
            registration_names=names,
            dependency_profile="enterprise" if set(names) & ENTERPRISE_LOGGER_NAMES else "required",
            behavioral_cases=next(
                (LOGGER_BEHAVIORAL_CASES[name] for name in names if name in LOGGER_BEHAVIORAL_CASES), ()
            ),
        )
        for names in grouped.values()
        if names
    }
    missing_optional: Final = ENTERPRISE_LOGGER_NAMES - registry.keys()
    unavailable: Final = {name: CoverageObligation(name, (name,), "enterprise", ()) for name in missing_optional}
    return MappingProxyType({**obligations, **unavailable})


LOGGER_OBLIGATIONS: Final = _logger_obligations()
GUARDRAIL_OBLIGATIONS: Final = MappingProxyType(
    {
        name: CoverageObligation(name, (name,), "required", GUARDRAIL_BEHAVIORAL_CASES.get(name, ()))
        for name in GUARDRAIL_NAMES | DISCOVERED_ONLY_GUARDRAIL_NAMES
    }
)
