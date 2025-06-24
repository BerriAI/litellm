"""
Registry mapping the callback class string to the class type.

This is used to get the class type from the callback class string.

Example:
    "datadog" -> DataDogLogger
    "prometheus" -> PrometheusLogger
"""

from litellm.integrations.agentops import AgentOps
from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook
from litellm.integrations.argilla import ArgillaLogger
from litellm.integrations.azure_storage.azure_storage import AzureBlobStorageLogger
from litellm.integrations.braintrust_logging import BraintrustLogger
from litellm.integrations.datadog.datadog import DataDogLogger
from litellm.integrations.datadog.datadog_llm_obs import DataDogLLMObsLogger
from litellm.integrations.deepeval import DeepEvalLogger
from litellm.integrations.galileo import GalileoObserve
from litellm.integrations.gcs_bucket.gcs_bucket import GCSBucketLogger
from litellm.integrations.gcs_pubsub.pub_sub import GcsPubSubLogger
from litellm.integrations.humanloop import HumanloopLogger
from litellm.integrations.lago import LagoLogger
from litellm.integrations.langfuse.langfuse_prompt_management import (
    LangfusePromptManagement,
)
from litellm.integrations.langsmith import LangsmithLogger
from litellm.integrations.literal_ai import LiteralAILogger
from litellm.integrations.mlflow import MlflowLogger
from litellm.integrations.openmeter import OpenMeterLogger
from litellm.integrations.opentelemetry import OpenTelemetry
from litellm.integrations.opik.opik import OpikLogger
from litellm.integrations.prometheus import PrometheusLogger
from litellm.integrations.s3_v2 import S3Logger
from litellm.integrations.vector_store_integrations.bedrock_vector_store import (
    BedrockVectorStore,
)
from litellm.proxy.hooks.dynamic_rate_limiter import _PROXY_DynamicRateLimitHandler


class CustomLoggerRegistry:
    """
    Registry mapping the callback class string to the class type.
    """
    CALLBACK_CLASS_STR_TO_CLASS_TYPE = {
        "lago": LagoLogger,
        "openmeter": OpenMeterLogger,
        "braintrust": BraintrustLogger,
        "galileo": GalileoObserve,
        "langsmith": LangsmithLogger,
        "literalai": LiteralAILogger,
        "prometheus": PrometheusLogger,
        "datadog": DataDogLogger,
        "datadog_llm_observability": DataDogLLMObsLogger,
        "gcs_bucket": GCSBucketLogger,
        "opik": OpikLogger,
        "argilla": ArgillaLogger,
        "opentelemetry": OpenTelemetry,
        "azure_storage": AzureBlobStorageLogger,
        "humanloop": HumanloopLogger,
        # OTEL compatible loggers
        "logfire": OpenTelemetry,
        "arize": OpenTelemetry,
        "langfuse_otel": OpenTelemetry,
        "arize_phoenix": OpenTelemetry,
        "langtrace": OpenTelemetry,
        "mlflow": MlflowLogger,
        "langfuse": LangfusePromptManagement,
        "otel": OpenTelemetry,
        "gcs_pubsub": GcsPubSubLogger,
        "anthropic_cache_control_hook": AnthropicCacheControlHook,
        "agentops": AgentOps,
        "bedrock_vector_store": BedrockVectorStore,
        "deepeval": DeepEvalLogger,
        "s3_v2": S3Logger,
        "dynamic_rate_limiter": _PROXY_DynamicRateLimitHandler,
    }

    try:
        from litellm_enterprise.enterprise_callbacks.generic_api_callback import (
            GenericAPILogger,
        )
        from litellm_enterprise.enterprise_callbacks.pagerduty.pagerduty import (
            PagerDutyAlerting,
        )
        from litellm_enterprise.enterprise_callbacks.send_emails.resend_email import (
            ResendEmailLogger,
        )
        from litellm_enterprise.enterprise_callbacks.send_emails.smtp_email import (
            SMTPEmailLogger,
        )

        enterprise_loggers = {
            "pagerduty": PagerDutyAlerting,
            "generic_api": GenericAPILogger,
            "resend_email": ResendEmailLogger,
            "smtp_email": SMTPEmailLogger,
        }
        CALLBACK_CLASS_STR_TO_CLASS_TYPE.update(enterprise_loggers)
    except ImportError:
        pass  # enterprise not installed

    @classmethod
    def get_callback_str_from_class_type(cls, class_type: type) -> str | None:
        """
        Get the callback string from the class type.
        
        Args:
            class_type: The class type to find the string for
            
        Returns:
            str: The callback string, or None if not found
        """
        for callback_str, callback_class in cls.CALLBACK_CLASS_STR_TO_CLASS_TYPE.items():
            if callback_class == class_type:
                return callback_str
        return None

    @classmethod
    def get_all_callback_strs_from_class_type(cls, class_type: type) -> list[str]:
        """
        Get all callback strings that map to the same class type.
        Some class types (like OpenTelemetry) have multiple string mappings.
        
        Args:
            class_type: The class type to find all strings for
            
        Returns:
            list: List of callback strings that map to the class type
        """
        callback_strs: list[str] = []
        for callback_str, callback_class in cls.CALLBACK_CLASS_STR_TO_CLASS_TYPE.items():
            if callback_class == class_type:
                callback_strs.append(callback_str)
        return callback_strs