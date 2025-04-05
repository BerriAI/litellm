from typing import Any, Callable, Dict, List, Literal, Optional, Union, get_args

from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.logging_callback_manager import LoggingCallbackManager

CALLBACK_TYPES = Union[str, Callable, CustomLogger]
input_callback: List[CALLBACK_TYPES] = []
success_callback: List[CALLBACK_TYPES] = []
failure_callback: List[CALLBACK_TYPES] = []
service_callback: List[CALLBACK_TYPES] = []
logging_callback_manager = LoggingCallbackManager()
_custom_logger_compatible_callbacks_literal = Literal[
    "lago",
    "openmeter",
    "logfire",
    "literalai",
    "dynamic_rate_limiter",
    "langsmith",
    "prometheus",
    "otel",
    "datadog",
    "datadog_llm_observability",
    "galileo",
    "braintrust",
    "arize",
    "arize_phoenix",
    "langtrace",
    "gcs_bucket",
    "azure_storage",
    "opik",
    "argilla",
    "mlflow",
    "langfuse",
    "pagerduty",
    "humanloop",
    "gcs_pubsub",
]
logged_real_time_event_types: Optional[Union[List[str], Literal["*"]]] = None
_known_custom_logger_compatible_callbacks: List = list(
    get_args(_custom_logger_compatible_callbacks_literal)
)
callbacks: List[
    Union[Callable, _custom_logger_compatible_callbacks_literal, CustomLogger]
] = []
langfuse_default_tags: Optional[List[str]] = None
langsmith_batch_size: Optional[int] = None
prometheus_initialize_budget_metrics: Optional[bool] = False
argilla_batch_size: Optional[int] = None
datadog_use_v1: Optional[bool] = False  # if you want to use v1 datadog logged payload
gcs_pub_sub_use_v1: Optional[bool] = (
    False  # if you want to use v1 gcs pubsub logged payload
)
argilla_transformation_object: Optional[Dict[str, Any]] = None
_async_input_callback: List[Union[str, Callable, CustomLogger]] = (
    []
)  # internal variable - async custom callbacks are routed here.
_async_success_callback: List[Union[str, Callable, CustomLogger]] = (
    []
)  # internal variable - async custom callbacks are routed here.
_async_failure_callback: List[Union[str, Callable, CustomLogger]] = (
    []
)  # internal variable - async custom callbacks are routed here.
pre_call_rules: List[Callable] = []
post_call_rules: List[Callable] = []
turn_off_message_logging: Optional[bool] = False
log_raw_request_response: bool = False
redact_messages_in_exceptions: Optional[bool] = False
redact_user_api_key_info: Optional[bool] = False
filter_invalid_headers: Optional[bool] = False
add_user_information_to_llm_headers: Optional[bool] = (
    None  # adds user_id, team_id, token hash (params from StandardLoggingMetadata) to request headers
)
store_audit_logs = False  # Enterprise feature, allow users to see audit logs
