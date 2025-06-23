"""
Registry mapping the callback class string to the class type.

This is used to get the class type from the callback class string.

Example:
    "datadog" -> DataDogLogger
    "prometheus" -> PrometheusLogger
"""

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

from litellm.integrations.agentops import AgentOps
from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook
from litellm.integrations.argilla import ArgillaLogger
from litellm.integrations.azure_storage.azure_storage import AzureBlobStorageLogger
from litellm.integrations.braintrust_logging import BraintrustLogger
from litellm.integrations.datadog.datadog import DataDogLogger
from litellm.integrations.datadog.datadog_llm_obs import DataDogLLMObsLogger
from litellm.integrations.deepeval import DeepEvalLogger
from litellm.integrations.deepeval.deepeval import DeepEvalLogger
from litellm.integrations.galileo import GalileoObserve
from litellm.integrations.gcs_bucket.gcs_bucket import GCSBucketLogger
from litellm.integrations.gcs_pubsub.pub_sub import GcsPubSubLogger
from litellm.integrations.humanloop import HumanloopLogger
from litellm.integrations.lago import LagoLogger
from litellm.integrations.langfuse.langfuse_otel import LangfuseOtelLogger
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
from litellm.integrations.vector_stores.bedrock_vector_store import BedrockVectorStore
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
        "pagerduty": PagerDutyAlerting,
        "gcs_pubsub": GcsPubSubLogger,
        "anthropic_cache_control_hook": AnthropicCacheControlHook,
        "agentops": AgentOps,
        "bedrock_vector_store": BedrockVectorStore,
        "generic_api": GenericAPILogger,
        "resend_email": ResendEmailLogger,
        "smtp_email": SMTPEmailLogger,
        "deepeval": DeepEvalLogger,
        "s3_v2": S3Logger,
        "langfuse_otel": OpenTelemetry,
    }
