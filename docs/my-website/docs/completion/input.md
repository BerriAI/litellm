import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Input Params

## Common Params 
LiteLLM accepts and translates the [OpenAI Chat Completion params](https://platform.openai.com/docs/api-reference/chat/create) across all providers. 

### Usage
```python
import litellm

# set env variables
os.environ["OPENAI_API_KEY"] = "your-openai-key"

## SET MAX TOKENS - via completion() 
response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{ "content": "Hello, how are you?","role": "user"}],
            max_tokens=10
        )

print(response)
```

### Translated OpenAI params

Use this function to get an up-to-date list of supported openai params for any model + provider. 

```python
from litellm import get_supported_openai_params

response = get_supported_openai_params(model="anthropic.claude-3", custom_llm_provider="bedrock")

print(response) # ["max_tokens", "tools", "tool_choice", "stream"]
```

This is a list of openai params we translate across providers.

Use `litellm.get_supported_openai_params()` for an updated list of params for each model + provider 

| Provider | temperature | max_completion_tokens | max_tokens | top_p | stream | stream_options | stop | n | presence_penalty | frequency_penalty | functions | function_call | logit_bias | user | response_format | seed| tools | tool_choice | logprobs | top_logprobs | extra_headers |
|--------------|-------------|------------------------|------------|-------|--------|----------------|------|-----|------------------|-------------------|-----------|----------------|-------------|------|------------------|-------------------|--------|--------------|----------|---------------|----------------------|
| Anthropic| ✅| ✅ | ✅ | ✅| ✅ | ✅ | ✅ | || | || | ✅ | ✅ | | ✅ | ✅ || | ✅|
| OpenAI | ✅| ✅ | ✅ | ✅| ✅ | ✅ | ✅ | ✅| ✅ | ✅| ✅| ✅ | ✅| ✅ | ✅ | ✅| ✅ | ✅ | ✅ | ✅| ✅|
| Azure OpenAI | ✅| ✅ | ✅ | ✅| ✅ | ✅ | ✅ | ✅| ✅ | ✅| ✅| ✅ | ✅| ✅ | ✅ | ✅| ✅ | ✅ | ✅ | ✅| ✅|
| xAI| ✅|| ✅ | ✅| ✅ | ✅ | ✅ | ✅| ✅ | ✅| || ✅| ✅ | ✅ | ✅| ✅ | ✅ | ✅ | ✅||
| Replicate| ✅| ✅ | ✅ | ✅| ✅ | ✅ || || | || ||| |||| ||
| Anyscale | ✅| ✅ | ✅ | ✅| ✅ | ✅ || || | || ||| |||| ||
| Cohere | ✅| ✅ | ✅ | ✅| ✅ | ✅ | ✅ | ✅|| | || ||| |||| ||
| Huggingface| ✅| ✅ | ✅ | ✅| ✅ | ✅ | ✅ | || | || ||| |||| ||
| Openrouter | ✅| ✅ | ✅ | ✅| ✅ | ✅ | ✅ | ✅| ✅ | ✅| ✅|| ||| ✅| ✅ ||| ||
| AI21 | ✅| ✅ | ✅ | ✅| ✅ | ✅ | ✅ | ✅|| | || ||| |||| ||
| VertexAI | ✅| ✅ | ✅ | | ✅ | ✅ || || | || || ✅ | ✅|||| ||
| Bedrock| ✅| ✅ | ✅ | ✅| ✅ | ✅ || || | || || ✅ (model dependent) | |||| ||
| Sagemaker| ✅| ✅ | ✅ | ✅| ✅ | ✅ | ✅ | || | || ||| |||| ||
| TogetherAI | ✅| ✅ | ✅ | ✅| ✅ | ✅ || || | ✅|| || ✅ | | ✅ | ✅ || ||
| Sambanova| ✅| ✅ | ✅ | ✅| ✅ | ✅ | ✅ | || | || || ✅ | | ✅ | ✅ || ||
| AlephAlpha | ✅| ✅ | ✅ | ✅| ✅ | ✅ | ✅ | || | || ||| |||| ||
| NLP Cloud| ✅| ✅ | ✅ | ✅| ✅ | ✅ || || | || ||| |||| ||
| Petals | ✅| ✅ || ✅| ✅ ||| || | || ||| |||| ||
| Ollama | ✅| ✅ | ✅ | ✅| ✅ | ✅ || ✅|| | || ✅||| | ✅ ||| ||
| Databricks | ✅| ✅ | ✅ | ✅| ✅ | ✅ || || | || ||| |||| ||
| ClarifAI | ✅| ✅ | ✅ | | ✅ | ✅ || || | || ||| |||| ||
| Github | ✅| ✅ | ✅ | ✅| ✅ | ✅ | ✅ | ✅| ✅ | ✅| ✅|| || ✅ | ✅ (model dependent) | ✅ (model dependent) || ||
| Novita AI| ✅| ✅ || ✅| ✅ | ✅ | ✅ | ✅| ✅ | ✅| || ✅||| |||| ||
| Bytez | ✅| ✅ || ✅| ✅ | | | ✅|| || || || || || ||
| OVHCloud AI Endpoints | ✅ | | ✅ | ✅ | ✅ | ✅ | ✅ | | | | | | | | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | |

:::note

By default, LiteLLM raises an exception if the openai param being passed in isn't supported. 

To drop the param instead, set `litellm.drop_params = True` or `completion(..drop_params=True)`.

This **ONLY DROPS UNSUPPORTED OPENAI PARAMS**. 

LiteLLM assumes any non-openai param is provider specific and passes it in as a kwarg in the request body

::: 

## Input Params

```python
def completion(
    model: str,
    messages: List = [],
    # Optional OpenAI params
    timeout: Optional[Union[float, int]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    n: Optional[int] = None,
    stream: Optional[bool] = None,
    stream_options: Optional[dict] = None,
    stop=None,
    max_completion_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[dict] = None,
    user: Optional[str] = None,
    # openai v1.0+ new params
    response_format: Optional[dict] = None,
    seed: Optional[int] = None,
    tools: Optional[List] = None,
    tool_choice: Optional[str] = None,
    parallel_tool_calls: Optional[bool] = None,
    logprobs: Optional[bool] = None,
    top_logprobs: Optional[int] = None,
    safety_identifier: Optional[str] = None,
    deployment_id=None,
    # soon to be deprecated params by OpenAI
    functions: Optional[List] = None,
    function_call: Optional[str] = None,
    # set api_base, api_version, api_key
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    api_key: Optional[str] = None,
    model_list: Optional[list] = None,  # pass in a list of api_base,keys, etc.
    # Optional liteLLM function params
    **kwargs,

) -> ModelResponse:
```
### Required Fields

- `model`: *string* - ID of the model to use. Refer to the model endpoint compatibility table for details on which models work with the Chat API.
  
- `messages`: *array* - A list of messages comprising the conversation so far.

#### Properties of `messages`
*Note* - Each message in the array contains the following properties:

- `role`: *string* - The role of the message's author. Roles can be: system, user, assistant, function or tool.

- `content`: *string or list[dict] or null* - The contents of the message. It is required for all messages, but may be null for assistant messages with function calls.

- `name`: *string (optional)* - The name of the author of the message. It is required if the role is "function". The name should match the name of the function represented in the content. It can contain characters (a-z, A-Z, 0-9), and underscores, with a maximum length of 64 characters.

- `function_call`: *object (optional)* - The name and arguments of a function that should be called, as generated by the model.

- `tool_call_id`: *str (optional)* - Tool call that this message is responding to.


[**See All Message Values**](https://github.com/BerriAI/litellm/blob/8600ec77042dacad324d3879a2bd918fc6a719fa/litellm/types/llms/openai.py#L392)

## Optional Fields

- `temperature`: *number or null (optional)* - The sampling temperature to be used, between 0 and 2. Higher values like 0.8 produce more random outputs, while lower values like 0.2 make outputs more focused and deterministic. 

- `top_p`: *number or null (optional)* - An alternative to sampling with temperature. It instructs the model to consider the results of the tokens with top_p probability. For example, 0.1 means only the tokens comprising the top 10% probability mass are considered.

- `n`: *integer or null (optional)* - The number of chat completion choices to generate for each input message.

- `stream`: *boolean or null (optional)* - If set to true, it sends partial message deltas. Tokens will be sent as they become available, with the stream terminated by a [DONE] message.

- `stream_options` *dict or null (optional)* - Options for streaming response. Only set this when you set `stream: true`

    - `include_usage` *boolean (optional)* - If set, an additional chunk will be streamed before the data: [DONE] message. The usage field on this chunk shows the token usage statistics for the entire request, and the choices field will always be an empty array. All other chunks will also include a usage field, but with a null value. 

- `stop`: *string/ array/ null (optional)* - Up to 4 sequences where the API will stop generating further tokens.

- `max_completion_tokens`: *integer (optional)* -  An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens.

- `max_tokens`: *integer (optional)* - The maximum number of tokens to generate in the chat completion.

- `presence_penalty`: *number or null (optional)* - It is used to penalize new tokens based on their existence in the text so far.

- `response_format`: *object (optional)* - An object specifying the format that the model must output.

    - Setting to `{ "type": "json_object" }` enables JSON mode, which guarantees the message the model generates is valid JSON.
    
    - Important: when using JSON mode, you must also instruct the model to produce JSON yourself via a system or user message. Without this, the model may generate an unending stream of whitespace until the generation reaches the token limit, resulting in a long-running and seemingly "stuck" request. Also note that the message content may be partially cut off if finish_reason="length", which indicates the generation exceeded max_tokens or the conversation exceeded the max context length.

- `seed`: *integer or null (optional)* - This feature is in Beta. If specified, our system will make a best effort to sample deterministically, such that repeated requests with the same seed and parameters should return the same result. Determinism is not guaranteed, and you should refer to the `system_fingerprint` response parameter to monitor changes in the backend.

- `tools`: *array (optional)* - A list of tools the model may call. Currently, only functions are supported as a tool. Use this to provide a list of functions the model may generate JSON inputs for.

    - `type`: *string* - The type of the tool. Currently, only function is supported.

    - `function`: *object* - Required.

- `tool_choice`: *string or object (optional)* - Controls which (if any) function is called by the model. none means the model will not call a function and instead generates a message. auto means the model can pick between generating a message or calling a function. Specifying a particular function via `{"type": "function", "function": {"name": "my_function"}}` forces the model to call that function.

    - `none` is the default when no functions are present. `auto` is the default if functions are present.

- `parallel_tool_calls`: *boolean (optional)* - Whether to enable parallel function calling during tool use. OpenAI default is true.

- `frequency_penalty`: *number or null (optional)* - It is used to penalize new tokens based on their frequency in the text so far.

- `logit_bias`: *map (optional)* - Used to modify the probability of specific tokens appearing in the completion.

- `user`: *string (optional)* - A unique identifier representing your end-user. This can help OpenAI to monitor and detect abuse.

- `timeout`: *int (optional)* - Timeout in seconds for completion requests (Defaults to 600 seconds)

- `logprobs`: * bool (optional)* - Whether to return log probabilities of the output tokens or not. If true returns the log probabilities of each output token returned in the content of message
        
- `top_logprobs`: *int (optional)* - An integer between 0 and 5 specifying the number of most likely tokens to return at each token position, each with an associated log probability. `logprobs` must be set to true if this parameter is used.

- `safety_identifier`: *string (optional)* - A unique identifier for tracking and managing safety-related requests. This parameter helps with safety monitoring and compliance tracking.

- `headers`: *dict (optional)* - A dictionary of headers to be sent with the request.

- `extra_headers`: *dict (optional)* - Alternative to `headers`, used to send extra headers in LLM API request. 

#### Deprecated Params
- `functions`: *array* - A list of functions that the model may use to generate JSON inputs. Each function should have the following properties:

    - `name`: *string* - The name of the function to be called. It should contain a-z, A-Z, 0-9, underscores and dashes, with a maximum length of 64 characters.
    
    - `description`: *string (optional)* - A description explaining what the function does. It helps the model to decide when and how to call the function.
    
    - `parameters`: *object* - The parameters that the function accepts, described as a JSON Schema object.
    
- `function_call`: *string or object (optional)* - Controls how the model responds to function calls.


#### litellm-specific params 

- `api_base`: *string (optional)* - The api endpoint you want to call the model with

- `api_version`: *string (optional)* - (Azure-specific) the api version for the call

- `num_retries`: *int (optional)* - The number of times to retry the API call if an APIError, TimeoutError or ServiceUnavailableError occurs 

- `context_window_fallback_dict`: *dict (optional)* - A mapping of model to use if call fails due to context window error

- `fallbacks`: *list (optional)* - A list of model names + params to be used, in case the initial call fails

- `metadata`: *dict (optional)* - Any additional data you want to be logged when the call is made (sent to logging integrations, eg. promptlayer and accessible via custom callback function)

**CUSTOM MODEL COST** 
- `input_cost_per_token`: *float (optional)* - The cost per input token for the completion call 

- `output_cost_per_token`: *float (optional)* - The cost per output token for the completion call 

**CUSTOM PROMPT TEMPLATE** (See [prompt formatting for more info](./prompt_formatting.md#format-prompt-yourself))
- `initial_prompt_value`: *string (optional)* - Initial string applied at the start of the input messages

- `roles`: *dict (optional)* - Dictionary specifying how to format the prompt based on the role + message passed in via `messages`. 

- `final_prompt_value`: *string (optional)* - Final string applied at the end of the input messages

- `bos_token`: *string (optional)* - Initial string applied at the start of a sequence

- `eos_token`: *string (optional)* - Initial string applied at the end of a sequence

- `hf_model_name`: *string (optional)* - [Sagemaker Only] The corresponding huggingface name of the model, used to pull the right chat template for the model. 

