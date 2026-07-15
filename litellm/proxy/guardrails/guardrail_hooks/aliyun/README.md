# Aliyun AI Guardrail Integration

This integration provides support for Alibaba Cloud's Content Security MultiModalGuard API in LiteLLM. It enables content moderation for text, images, and MCP tool calls with configurable protection levels and streaming support.

## Features

- Support for Aliyun Content Security MultiModalGuard API (Version 2022-03-02)
- HMAC-SHA1 request signing
- Input moderation (pre-call): scans text and images from the last consecutive user messages
- Output moderation (post-call): full response scan for non-streaming; sliding window detection for streaming
- MCP tool call moderation (pre/post MCP): inspects tool name + arguments and tool execution results
- Multi-modal detection: supports mixed text + HTTP(S) image URL inspection
- Long text chunking: automatic splitting at punctuation boundaries with concurrent chunk verification
- Four protection levels: low / medium / high / max
- Six detection types: contentModeration, sensitiveData, promptAttack, maliciousUrl, modelHallucination, customLabel
- Configurable per-region endpoint routing

## Configuration

### Required Parameters

- `access_key_id`: Alibaba Cloud Access Key ID (supports `os.environ/` prefix for environment variable reference, e.g. `os.environ/ACCESS_KEY_ID`)
- `access_key_secret`: Alibaba Cloud Access Key Secret (supports `os.environ/` prefix for environment variable reference, e.g. `os.environ/ACCESS_KEY_SECRET`)

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `level` | str | `medium` | Protection level: low / medium / high / max |
| `max_text_length` | int | `2000` | Maximum text length per API call |
| `stream_window_size` | int | `500` | Streaming check window size (characters) |
| `stream_slide_step` | int | `300` | Streaming check slide step (characters) |
| `stream_first_check_step` | int | `50` | First-byte check threshold (characters) |
| `region_id` | str | `cn-shanghai` | Alibaba Cloud region ID |
| `service_input` | str | `query_security_check_pro` | Service code for input detection |
| `service_output` | str | `response_security_check_pro` | Service code for output detection |
| `service_mcp` | str | `query_security_check_pro` | Service code for MCP tool detection |

## Usage Examples

### Example 1: Config YAML (Pre-call + Post-call)

```yaml
model_list:
  - model_name: gpt-4
    litellm_params:
      model: openai/gpt-4
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: aliyun-guard
    litellm_params:
      guardrail: aliyun_ai_guardrail
      mode: [pre_call, post_call]
      default_on: true
      access_key_id: os.environ/ACCESS_KEY_ID
      access_key_secret: os.environ/ACCESS_KEY_SECRET
      level: medium
      max_text_length: 2000
      stream_window_size: 500
      stream_slide_step: 300
      stream_first_check_step: 50
      region_id: cn-shanghai
      service_input: query_security_check_pro
      service_output: response_security_check_pro
      service_mcp: query_security_check_pro
```

### Example 2: Per-Request Usage

```bash
curl --location 'http://localhost:4000/chat/completions' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}],
    "guardrails": ["aliyun-guard"]
}'
```

### Example 3: Multi-modal Input (Text + Image)

```bash
curl --location 'http://localhost:4000/chat/completions' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "model": "gpt-4",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
            ]
        }
    ],
    "guardrails": ["aliyun-guard"]
}'
```

## API Endpoint

- **URL**: `https://green-cip.{region_id}.aliyuncs.com/`
- **Method**: POST
- **Content-Type**: `application/x-www-form-urlencoded`
- **Action**: `MultiModalGuard`
- **Version**: `2022-03-02`
- **Signature Method**: HMAC-SHA1

### Request Format

The `ServiceParameters` field is a JSON string with the following structure:

**Text only:**
```json
{"content": "text to check..."}
```

**Images only:**
```json
{"imageUrls": ["https://example.com/img1.png", "https://example.com/img2.png"]}
```

**Mixed (text + images):**
```json
{"content": "text to check...", "imageUrls": ["https://example.com/img1.png"]}
```

## Response Format

```json
{
  "RequestId": "xxx",
  "Code": 200,
  "Data": {
    "Suggestion": "block",
    "Detail": [
      {
        "Type": "contentModeration",
        "Suggestion": "block",
        "Result": [
          {
            "Label": "violence",
            "Confidence": 99.5,
            "RiskLevel": "high"
          }
        ]
      }
    ]
  }
}
```

## Supported Regions

| Region ID | Location |
|-----------|----------|
| `cn-shanghai` | China (Shanghai) - Default |
| `cn-beijing` | China (Beijing) |
| `cn-hangzhou` | China (Hangzhou) |
| `cn-shenzhen` | China (Shenzhen) |
| `cn-chengdu` | China (Chengdu) |
| `ap-southeast-1` | Singapore |
| `eu-central-1` | Germany (Frankfurt) |

## Protection Levels

| Level | Behavior | Description |
|-------|----------|-------------|
| `low` | Blocks low / medium / high risk | Most strict — blocks all risk levels |
| `medium` | Blocks medium / high risk | Default — balanced protection |
| `high` | Blocks high risk only | Permissive — only blocks high-severity content |
| `max` | Observe mode, no blocking | Logging only — no content is blocked |

## Supported Event Hooks

- `pre_call`: Input text + image detection before LLM API call
- `post_call`: Output detection after LLM API call (non-streaming: full response; streaming: sliding window)
- `pre_mcp_call`: MCP tool argument detection before tool execution
- `post_mcp_call`: MCP tool result detection after tool execution

## Error Handling

- **Input violation (pre-call)**: Raises `HTTPException(status_code=400)` with detection type, risk level, and detailed results
- **Output violation (post-call, non-streaming)**: Raises `HTTPException(status_code=400)` with violation details
- **Streaming violation (post-call, streaming)**: Yields an SSE error event (does NOT raise an exception)

## Environment Variables

You can use any environment variable name and reference it in `config.yaml` via the `os.environ/YOUR_VAR_NAME` format. It is recommended to use dedicated variable names (e.g. `ACCESS_KEY_ID`, `ACCESS_KEY_SECRET`) to avoid conflicts with other service credentials.

Example:
```bash
export ACCESS_KEY_ID="your-access-key-id"
export ACCESS_KEY_SECRET="your-access-key-secret"
```

## Detection Types

| Type | Description |
|------|-------------|
| `contentModeration` | General content safety (violence, porn, etc.) |
| `sensitiveData` | Sensitive/private data detection |
| `promptAttack` | Prompt injection and jailbreak detection |
| `maliciousUrl` | Malicious URL detection |
| `modelHallucination` | Model hallucination detection |
| `customLabel` | Custom label-based detection |

## Notes

- Only the last consecutive block of user messages is inspected; system/assistant/tool messages are not checked
- Only HTTP(S) publicly accessible image URLs are supported; base64 inline images are not supported
- Text chunking splits preferentially at punctuation to preserve semantic integrity
- Streaming moderation uses a buffer-and-release mechanism: all chunks are buffered until the check passes
- The guardrail inherits from both `AliyunGuardrailBase` and `CustomGuardrail`
- The guardrail enum value is `ALIYUN_AI_GUARDRAIL = "aliyun_ai_guardrail"`

## References

- [Alibaba Cloud Content Security Documentation](https://help.aliyun.com/zh/content-moderation/)
- [LiteLLM Guardrails Documentation](https://docs.litellm.ai/docs/proxy/guardrails/quick_start)
