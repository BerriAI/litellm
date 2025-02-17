# Request Headers

Special headers that are supported by LiteLLM.

## LiteLLM Headers

`x-litellm-timeout` Optional[float]: The timeout for the request in seconds.

`x-litellm-enable-message-redaction`: Optional[bool]: Don't log the message content to logging integrations. Just track spend. [Learn More](./logging#redact-messages-response-content)

## Anthropic Headers

`anthropic-version` Optional[str]: The version of the Anthropic API to use.  
`anthropic-beta` Optional[str]: The beta version of the Anthropic API to use.

## OpenAI Headers

`openai-organization` Optional[str]: The organization to use for the OpenAI API. (currently needs to be enabled via `general_settings::forward_openai_org_id: true`)



