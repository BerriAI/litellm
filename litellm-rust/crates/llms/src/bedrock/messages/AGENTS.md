# structure

- Bedrock Messages endpoints, authentication policy, wire adaptation and response decoding

# boundaries

- Implement `base_llm/messages` contracts and consume `llms-types::formats::messages`; orchestration belongs in `inference-messages`
- Reuse Anthropic payload helpers explicitly when applicable to Claude, keeping Bedrock differences here
- Shared provider helpers do not make Anthropic policy a format-wide default
