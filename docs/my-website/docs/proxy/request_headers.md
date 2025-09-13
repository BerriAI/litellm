# Request Headers

Special headers that are supported by LiteLLM.

## Header Forwarding

By default, LiteLLM does not forward client headers to LLM provider APIs. However, you can selectively enable header forwarding for specific model groups. [Learn more about configuring header forwarding](./forward_client_headers.md).

## LiteLLM Headers

`x-litellm-timeout` Optional[float]: The timeout for the request in seconds.

`x-litellm-stream-timeout` Optional[float]: The timeout for getting the first chunk of the response in seconds (only applies for streaming requests). [Demo Video](https://www.loom.com/share/8da67e4845ce431a98c901d4e45db0e5)

`x-litellm-enable-message-redaction`: Optional[bool]: Don't log the message content to logging integrations. Just track spend. [Learn More](./logging#redact-messages-response-content)

`x-litellm-tags`: Optional[str]: A comma separated list (e.g. `tag1,tag2,tag3`) of tags to use for [tag-based routing](./tag_routing) **OR** [spend-tracking](./enterprise.md#tracking-spend-for-custom-tags).

`x-litellm-num-retries`: Optional[int]: The number of retries for the request.

`x-litellm-spend-logs-metadata`: Optional[str]: JSON string containing custom metadata to include in spend logs. Example: `{"user_id": "12345", "project_id": "proj_abc", "request_type": "chat_completion"}`. [Learn More](../proxy/enterprise#tracking-spend-with-custom-metadata)

## Anthropic Headers

`anthropic-version` Optional[str]: The version of the Anthropic API to use.  
`anthropic-beta` Optional[str]: The beta version of the Anthropic API to use.
    - For `/v1/messages` endpoint, this will always be forward the header to the underlying model.
    - For `/chat/completions` endpoint, this will only be forwarded if the model is configured in `forward_client_headers_to_llm_api`. [Learn more](./forward_client_headers.md)

## OpenAI Headers

`openai-organization` Optional[str]: The organization to use for the OpenAI API. (currently needs to be enabled via `general_settings::forward_openai_org_id: true`)

## Custom Headers

Custom headers starting with `x-` can be forwarded to LLM provider APIs when the model is configured in `forward_client_headers_to_llm_api`. [Learn more about header forwarding configuration](./forward_client_headers.md).



