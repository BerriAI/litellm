
# StandardLoggingPayload Specification

Found under `kwargs["standard_logging_object"]`. This is a standard payload, logged for every successful and failed response.

## StandardLoggingPayload

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique identifier |
| `trace_id` | `str` | Trace multiple LLM calls belonging to same overall request |
| `call_type` | `str` | Type of call |
| `response_cost` | `float` | Cost of the response in USD ($) |
| `cost_breakdown` | `Optional[CostBreakdown]` | Detailed cost breakdown object |
| `response_cost_failure_debug_info` | `StandardLoggingModelCostFailureDebugInformation` | Debug information if cost tracking fails |
| `status` | `StandardLoggingPayloadStatus` | Status of the payload |
| `status_fields` | `StandardLoggingPayloadStatusFields` | Typed status fields for easy filtering and analytics |
| `total_tokens` | `int` | Total number of tokens |
| `prompt_tokens` | `int` | Number of prompt tokens |
| `completion_tokens` | `int` | Number of completion tokens |
| `startTime` | `float` | Start time of the call |
| `endTime` | `float` | End time of the call |
| `completionStartTime` | `float` | Time to first token for streaming requests |
| `response_time` | `float` | Total response time. If streaming, this is the time to first token |
| `model_map_information` | `StandardLoggingModelInformation` | Model mapping information |
| `model` | `str` | Model name sent in request |
| `model_id` | `Optional[str]` | Model ID of the deployment used |
| `model_group` | `Optional[str]` | `model_group` used for the request |
| `api_base` | `str` | LLM API base URL |
| `metadata` | `StandardLoggingMetadata` | Metadata information |
| `cache_hit` | `Optional[bool]` | Whether cache was hit |
| `cache_key` | `Optional[str]` | Optional cache key |
| `saved_cache_cost` | `float` | Cost saved by cache |
| `request_tags` | `list` | List of request tags |
| `end_user` | `Optional[str]` | Optional end user identifier |
| `requester_ip_address` | `Optional[str]` | Optional requester IP address |
| `messages` | `Optional[Union[str, list, dict]]` | Messages sent in the request |
| `response` | `Optional[Union[str, list, dict]]` | LLM response |
| `error_str` | `Optional[str]` | Optional error string |
| `error_information` | `Optional[StandardLoggingPayloadErrorInformation]` | Optional error information |
| `model_parameters` | `dict` | Model parameters |
| `hidden_params` | `StandardLoggingHiddenParams` | Hidden parameters |

## Cost Breakdown

The `cost_breakdown` field provides detailed cost breakdown for completion requests as a `CostBreakdown` object containing:

- **`input_cost`**: Cost of input/prompt tokens including cache creation tokens
- **`output_cost`**: Cost of output/completion tokens (including reasoning tokens if applicable)
- **`tool_usage_cost`**: Cost of built-in tools usage (e.g., web search, code interpreter)
- **`total_cost`**: Total cost of input + output + tool usage

**Note**: This field is populated for all call types. For non-completion calls, `input_cost` and `output_cost` may be 0.

The total cost relationship is: `response_cost = cost_breakdown.total_cost`

### CostBreakdown Type

```python
class CostBreakdown(TypedDict, total=False):
    input_cost: float        # Cost of input/prompt tokens in USD
    output_cost: float       # Cost of output/completion tokens in USD (includes reasoning)
    tool_usage_cost: float   # Cost of built-in tools usage in USD
    total_cost: float        # Total cost in USD
```

## StandardLoggingUserAPIKeyMetadata

| Field | Type | Description |
|-------|------|-------------|
| `user_api_key_hash` | `Optional[str]` | Hash of the litellm virtual key |
| `user_api_key_alias` | `Optional[str]` | Alias of the API key |
| `user_api_key_org_id` | `Optional[str]` | Organization ID associated with the key |
| `user_api_key_team_id` | `Optional[str]` | Team ID associated with the key |
| `user_api_key_user_id` | `Optional[str]` | User ID associated with the key |
| `user_api_key_team_alias` | `Optional[str]` | Team alias associated with the key |

## StandardLoggingMetadata

Inherits from `StandardLoggingUserAPIKeyMetadata` and adds:

| Field | Type | Description |
|-------|------|-------------|
| `spend_logs_metadata` | `Optional[dict]` | Key-value pairs for spend logging |
| `requester_ip_address` | `Optional[str]` | Requester's IP address |
| `requester_metadata` | `Optional[dict]` | Additional requester metadata |
| `vector_store_request_metadata` | `Optional[List[StandardLoggingVectorStoreRequest]]` | Vector store request metadata |
| `requester_custom_headers` | Dict[str, str] | Any custom (`x-`) headers sent by the client to the proxy. |
| `prompt_management_metadata` | `Optional[StandardLoggingPromptManagementMetadata]` | Prompt management and versioning metadata |
| `mcp_tool_call_metadata` | `Optional[StandardLoggingMCPToolCall]` | MCP (Model Context Protocol) tool call information and cost tracking |
| `applied_guardrails` | `Optional[List[str]]` | List of applied guardrail names |
| `usage_object` | `Optional[dict]` | Raw usage object from the LLM provider |
| `cold_storage_object_key` | `Optional[str]` | S3/GCS object key for cold storage retrieval |
| `guardrail_information` | `Optional[list[StandardLoggingGuardrailInformation]]` | Guardrail information |


## StandardLoggingVectorStoreRequest

| Field | Type | Description |
|-------|------|-------------|
| vector_store_id | Optional[str] | ID of the vector store |
| custom_llm_provider | Optional[str] | Custom LLM provider the vector store is associated with (e.g., bedrock, openai, anthropic) |
| query | Optional[str] | Query to the vector store |
| vector_store_search_response | Optional[VectorStoreSearchResponse] | OpenAI format vector store search response |
| start_time | Optional[float] | Start time of the vector store request |
| end_time | Optional[float] | End time of the vector store request |


## StandardLoggingAdditionalHeaders

| Field | Type | Description |
|-------|------|-------------|
| `x_ratelimit_limit_requests` | `int` | Rate limit for requests |
| `x_ratelimit_limit_tokens` | `int` | Rate limit for tokens |
| `x_ratelimit_remaining_requests` | `int` | Remaining requests in rate limit |
| `x_ratelimit_remaining_tokens` | `int` | Remaining tokens in rate limit |

## StandardLoggingHiddenParams

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | `Optional[str]` | Optional model ID |
| `cache_key` | `Optional[str]` | Optional cache key |
| `api_base` | `Optional[str]` | Optional API base URL |
| `response_cost` | `Optional[str]` | Optional response cost |
| `additional_headers` | `Optional[StandardLoggingAdditionalHeaders]` | Additional headers |
| `batch_models` | `Optional[List[str]]` | Only set for Batches API. Lists the models used for cost calculation |
| `litellm_model_name` | `Optional[str]` | Model name sent in request |

## StandardLoggingModelInformation

| Field | Type | Description |
|-------|------|-------------|
| `model_map_key` | `str` | Model map key |
| `model_map_value` | `Optional[ModelInfo]` | Optional model information |

## StandardLoggingModelCostFailureDebugInformation

| Field | Type | Description |
|-------|------|-------------|
| `error_str` | `str` | Error string |
| `traceback_str` | `str` | Traceback string |
| `model` | `str` | Model name |
| `cache_hit` | `Optional[bool]` | Whether cache was hit |
| `custom_llm_provider` | `Optional[str]` | Optional custom LLM provider |
| `base_model` | `Optional[str]` | Optional base model |
| `call_type` | `str` | Call type |
| `custom_pricing` | `Optional[bool]` | Whether custom pricing was used |

## StandardLoggingPayloadErrorInformation

| Field | Type | Description |
|-------|------|-------------|
| `error_code` | `Optional[str]` | Optional error code (eg. "429") |
| `error_class` | `Optional[str]` | Optional error class (eg. "RateLimitError") |
| `llm_provider` | `Optional[str]` | LLM provider that returned the error (eg. "openai")` |

## StandardLoggingPayloadStatus

A literal type with two possible values:
- `"success"`
- `"failure"`

## StandardLoggingGuardrailInformation

| Field                 | Type | Description                                                                                                                                                               |
|-----------------------|------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `guardrail_name`      | `Optional[str]` | Guardrail name                                                                                                                                                            |
| `guardrail_provider`  | `Optional[str]` | Guardrail provider                                                                                                                                                        |
| `guardrail_mode`      | `Optional[Union[GuardrailEventHooks, List[GuardrailEventHooks]]]` | Guardrail mode                                                                                                                                                            |
| `guardrail_request`   | `Optional[dict]` | Guardrail request                                                                                                                                                         |
| `guardrail_response`  | `Optional[Union[dict, str, List[dict]]]` | Guardrail response                                                                                                                                                        |
| `guardrail_status`    | `Literal["success", "guardrail_intervened", "guardrail_failed_to_respond"]` | Guardrail execution status: `success` = no violations detected, `blocked` = content blocked/modified due to policy violations, `failure` = technical error or API failure |
| `start_time`          | `Optional[float]` | Start time of the guardrail                                                                                                                                               |
| `end_time`            | `Optional[float]` | End time of the guardrail                                                                                                                                                 |
| `duration`            | `Optional[float]` | Duration of the guardrail in seconds                                                                                                                                      |
| `masked_entity_count` | `Optional[Dict[str, int]]` | Count of masked entities                                                                                                                                                  |

## StandardLoggingPayloadStatusFields

Typed status fields for easy filtering and analytics.

| Field | Type | Description |
|-------|------|-------------|
| `llm_api_status` | `StandardLoggingPayloadStatus` | Status of the LLM API call: `"success"` if completed successfully, `"failure"` if errored |
| `guardrail_status` | `GuardrailStatus` | Status of guardrail execution (see below) |

### StandardLoggingPayloadStatus

A literal type with two possible values:
- `"success"` - The LLM API request completed successfully
- `"failure"` - The LLM API request failed

### GuardrailStatus

A literal type with four possible values:
- `"success"` - Guardrail ran and allowed content through (no violations detected)
- `"guardrail_intervened"` - Guardrail blocked or modified content due to policy violations
- `"guardrail_failed_to_respond"` - Guardrail had a technical failure or API error
- `"not_run"` - No guardrail was executed for this request

### Usage Examples

Filter logs for requests where guardrails intervened:
```json
{
  "status_fields": {
    "guardrail_status": "guardrail_intervened"
  }
}
```

Find guardrail technical failures:
```json
{
  "status_fields": {
    "guardrail_status": "guardrail_failed_to_respond"
  }
}
```

Get successful LLM requests:
```json
{
  "status_fields": {
    "llm_api_status": "success"
  }
}
```

Find requests where guardrails ran successfully without intervention:
```json
{
  "status_fields": {
    "guardrail_status": "success",
    "llm_api_status": "success"
  }
}
```

Find requests where no guardrail was run:
```json
{
  "status_fields": {
    "guardrail_status": "not_run"
  }
}
```

## StandardLoggingPromptManagementMetadata

Used for tracking prompt versioning and management information.

| Field | Type | Description |
|-------|------|-------------|
| `prompt_id` | `str` | **Required**. Unique identifier for the prompt template or version |
| `prompt_variables` | `Optional[dict]` | Variables/parameters used in the prompt template (e.g., `{"user_name": "John", "context": "support"}`) |
| `prompt_integration` | `str` | **Required**. Integration or system managing the prompt (e.g., `"langfuse"`, `"promptlayer"`, `"custom"`) |

## StandardLoggingMCPToolCall

Used to track Model Context Protocol (MCP) tool calls within LiteLLM requests. This provides detailed logging for external tool integrations.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | **Required**. The name of the tool being called (e.g., `"get_weather"`, `"search_database"`) |
| `arguments` | `dict` | **Required**. Arguments passed to the tool as key-value pairs |
| `result` | `Optional[dict]` | The response/result returned by the tool execution (populated by custom logging hooks) |
| `mcp_server_name` | `Optional[str]` | Name of the MCP server that handled the tool call (e.g., `"weather-service"`, `"database-connector"`) |
| `mcp_server_logo_url` | `Optional[str]` | URL for the MCP server's logo (used for UI display in LiteLLM dashboard) |
| `namespaced_tool_name` | `Optional[str]` | Fully qualified tool name including server prefix (e.g., `"deepwiki-mcp/get_page_content"`, `"github-mcp/create_issue"`) |
| `mcp_server_cost_info` | `Optional[MCPServerCostInfo]` | Cost tracking information for the tool call |

### MCPServerCostInfo

Cost tracking structure for MCP server tool calls:

| Field | Type | Description |
|-------|------|-------------|
| `default_cost_per_query` | `Optional[float]` | Default cost in USD for any tool call to this MCP server |
| `tool_name_to_cost_per_query` | `Optional[Dict[str, float]]` | Per-tool cost mapping for granular pricing (e.g., `{"search": 0.01, "create": 0.05}`) |

### Usage

```python
# Basic MCP tool call metadata
mcp_tool_call = {
    "name": "search_documents",
    "arguments": {
        "query": "machine learning tutorials",
        "limit": 10,
        "filter": "type:pdf"
    },
    "mcp_server_name": "document-search-service",
    "namespaced_tool_name": "docs-mcp/search_documents",
    "mcp_server_cost_info": {
        "default_cost_per_query": 0.02,
        "tool_name_to_cost_per_query": {
            "search_documents": 0.02,
            "get_document": 0.01
        }
    }
}

# optional result field (via custom logging hooks)
mcp_tool_call_with_result = {
    "name": "search_documents",
    "arguments": {
        "query": "machine learning tutorials",
        "limit": 10,
        "filter": "type:pdf"
    },
    "result": {
        "documents": [...],
        "total_found": 42,
        "search_time_ms": 150
    },
    "mcp_server_name": "document-search-service",
    "namespaced_tool_name": "docs-mcp/search_documents",
    "mcp_server_cost_info": {
        "default_cost_per_query": 0.02,
        "tool_name_to_cost_per_query": {
            "search_documents": 0.02,
            "get_document": 0.01
        }
    }
}
```