"""
Types and field definitions for router settings management endpoints
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class RouterSettingsField(BaseModel):
    field_name: str
    field_type: str
    field_value: Any
    field_description: str
    field_default: Any = None
    options: Optional[List[str]] = None  # For fields with predefined options/enum values
    ui_field_name: str  # User-friendly display name
    link: Optional[str] = None  # Documentation link for the field


# Routing strategy descriptions
ROUTING_STRATEGY_DESCRIPTIONS: Dict[str, str] = {
    "simple-shuffle": "Randomly picks a deployment from the list. Simple and fast.",
    "least-busy": "Routes to the deployment with the lowest number of ongoing requests.",
    "latency-based-routing": "Routes to the deployment with the lowest latency over a sliding window.",
    "cost-based-routing": "Routes to the deployment with the lowest cost per token.",
    "usage-based-routing": "Routes to the deployment with the lowest TPM (Tokens Per Minute) usage. (deprecated)",
    "usage-based-routing-v2": "Improved version of usage-based routing with better tracking.",
}


# Define all available router settings fields
ROUTER_SETTINGS_FIELDS: List[RouterSettingsField] = [
    RouterSettingsField(
        field_name="routing_strategy",
        field_type="String",
        field_value=None,
        field_description="Routing strategy to use for load balancing across deployments",
        field_default="simple-shuffle",
        options=[],  # Will be populated dynamically from Router class
        ui_field_name="Routing Strategy",
    ),
    RouterSettingsField(
        field_name="routing_strategy_args",
        field_type="Dictionary",
        field_value=None,
        field_description="Arguments to pass to the routing strategy (e.g., ttl, lowest_latency_buffer for latency-based-routing)",
        field_default={},
        ui_field_name="Routing Strategy Args",
    ),
    RouterSettingsField(
        field_name="num_retries",
        field_type="Integer",
        field_value=None,
        field_description="Number of retries for failed requests",
        field_default=0,
        ui_field_name="Number of Retries",
    ),
    RouterSettingsField(
        field_name="timeout",
        field_type="Float",
        field_value=None,
        field_description="Timeout for requests in seconds",
        field_default=None,
        ui_field_name="Timeout",
    ),
    RouterSettingsField(
        field_name="stream_timeout",
        field_type="Float",
        field_value=None,
        field_description="Timeout for streaming requests in seconds",
        field_default=None,
        ui_field_name="Stream Timeout",
    ),
    RouterSettingsField(
        field_name="max_fallbacks",
        field_type="Integer",
        field_value=None,
        field_description="Maximum number of fallbacks to try before exiting the call",
        field_default=5,
        ui_field_name="Max Fallbacks",
    ),
    RouterSettingsField(
        field_name="fallbacks",
        field_type="List",
        field_value=None,
        field_description="List of fallback model mappings",
        field_default=[],
        ui_field_name="Fallbacks",
    ),
    RouterSettingsField(
        field_name="context_window_fallbacks",
        field_type="List",
        field_value=None,
        field_description="List of fallback models for context window errors",
        field_default=[],
        ui_field_name="Context Window Fallbacks",
    ),
    RouterSettingsField(
        field_name="content_policy_fallbacks",
        field_type="List",
        field_value=None,
        field_description="List of fallback models for content policy errors",
        field_default=[],
        ui_field_name="Content Policy Fallbacks",
    ),
    RouterSettingsField(
        field_name="allowed_fails",
        field_type="Integer",
        field_value=None,
        field_description="Number of times a deployment can fail before being added to cooldown",
        field_default=None,
        ui_field_name="Allowed Fails",
    ),
    RouterSettingsField(
        field_name="cooldown_time",
        field_type="Float",
        field_value=None,
        field_description="Time in seconds to cooldown a deployment after failure",
        field_default=None,
        ui_field_name="Cooldown Time",
    ),
    RouterSettingsField(
        field_name="retry_after",
        field_type="Integer",
        field_value=None,
        field_description="Minimum time to wait before retrying a failed request in seconds",
        field_default=0,
        ui_field_name="Retry After",
    ),
    RouterSettingsField(
        field_name="retry_policy",
        field_type="Dictionary",
        field_value=None,
        field_description="Custom retry policy for different exception types",
        field_default=None,
        ui_field_name="Retry Policy",
    ),
    RouterSettingsField(
        field_name="model_group_alias",
        field_type="Dictionary",
        field_value=None,
        field_description="Aliases for model groups",
        field_default={},
        ui_field_name="Model Group Alias",
    ),
    RouterSettingsField(
        field_name="enable_pre_call_checks",
        field_type="Boolean",
        field_value=None,
        field_description="Enable pre-call checks before routing requests",
        field_default=False,
        ui_field_name="Enable Pre-call Checks",
    ),
    RouterSettingsField(
        field_name="default_litellm_params",
        field_type="Dictionary",
        field_value=None,
        field_description="Default parameters for Router.chat.completion.create",
        field_default=None,
        ui_field_name="Default LiteLLM Params",
    ),
    RouterSettingsField(
        field_name="set_verbose",
        field_type="Boolean",
        field_value=None,
        field_description="Enable verbose logging for router",
        field_default=False,
        ui_field_name="Verbose Logging",
    ),
    RouterSettingsField(
        field_name="default_max_parallel_requests",
        field_type="Integer",
        field_value=None,
        field_description="Default maximum parallel requests across all deployments",
        field_default=None,
        ui_field_name="Max Parallel Requests",
    ),
    RouterSettingsField(
        field_name="enable_tag_filtering",
        field_type="Boolean",
        field_value=None,
        field_description="Enable tag-based routing to route requests based on tags",
        field_default=False,
        ui_field_name="Enable Tag Filtering",
        link="https://docs.litellm.ai/docs/proxy/tag_routing",
    ),
    RouterSettingsField(
        field_name="disable_cooldowns",
        field_type="Boolean",
        field_value=None,
        field_description="Disable cooldown mechanism for failed deployments",
        field_default=None,
        ui_field_name="Disable Cooldowns",
    ),
]

