# Overview
This spec is not expected to be extremely detailed for the developer. The spec assumes the developer will take the 
time when building their development plan to review how to implement each step correctly within LiteLLM. As the designer,
you will look at the existing callback extensions and answer the specific details for where each piece of data is 
available from the arguments provided to the callback methods.

The goal is to create an callback extension in order to send the AI data to New Relic.
The extension should be created in `litellm/integrations/newrelic/`. There are several
existing examples that can be reviewed and I would recommend the `litellm/integrations/opentelemetry.py`
as an example.

A user should be able to enabled the new callback with the following LiteLLM configuration:

```
    litellm.callbacks = ["newrelic"]
```

In addition, the user would have to provide their New Relic license key and app name via environment variables. If these
environment variables are not set, then the New Relic callback should return early (no-op) and not execute the rest of 
the callback logic. Returning early should not raise an exception.

```
    export NEW_RELIC_LICENSE_KEY="your_license_key"
    export NEW_RELIC_APP_NAME="your_app_name"
```

The New Relic callback will be implemented by extending the class `CustomLogger`. This class implements a number of methods,
and the approach for each is defined below.

* `log_pre_api_call` - Unused, do nothing.
* `log_post_api_call` - Unused, do nothing.
* `log_success_event` - Implement as the main success path for non-streaming requests.
* `log_failure_event` - Implement to log an error metric to New Relic
* `async_log_success_event` - Implement as the main success path for async streaming requests.
* `async_log_failure_event` - Implement to log an error metric to New Relic

## New Relic Configuration
There are a number of configuration options available to New Relic. The important ones for our test are already defined
in the `.env` file. Just use the CLI to set all of these as actual env vars such as `set -a; source .env; set +a`.

If the `NEW_RELIC_AI_MONITORING_RECORD_CONTENT_ENABLED` is not set or is not set to true, generate the events, but 
do not store the `content` of the `LlmChatCompletionMessage` event. This is important for PII and data security. 
The value of this environment variable can be either `true` or `'true'` (string). This configuration applies to *all*
messages (e.g. context, user, assistant, tools, etc).

## LLM sync vs async
LiteLLM callbacks are defined by the CustomLogger class. The methods and definition are defined above and which methods
we need actual implementations for. The `log_success_event` method is for synchronous LLM calls and the
`async_log_success_event` method is for asynchronous LLM calls. Both methods should have the same logic for sending
the New Relic events. In each case, when processing LLM messages, the entire message will already be provided by LiteLLM.

## LiteLLM errors
If LiteLLM invokes an error handling callback, do not send the New Relic AI events. Instead, generate a New Relic
Metric named `LLM/LiteLLM/Error` with a count of 1. 

## Dev Plan Notes
In the dev plan, when referring to New Relic data, use the actual methods from the New Relic Python agent. Do NOT use the
JSON examples that were included in the specification as JSON was used just as an illustration and easy to type.

## Types of LLM data
For now, assume that only text-based LLM conversations will be used. New Relic does not support images, audio, video, or
any other type of media outside of text.

# New Relic Python Agent
The customer will need to install the New Relic Python agent with their application. This can be done with standard
python tooling such as pip, poetry, or uv. The customer should install a minimum Python `newrelic` library of `11.0.1`.
For this use case, I would recommend using the `newrelic-admin` 
[installation method](https://docs.newrelic.com/docs/apm/agents/python-agent/installation/python-agent-admin-script-advanced-usage/) 
to be able to have the New Relic agent wrap the application versus trying to install the agent into the application.

# New Relic AI Event Model
The following New Relic events should be created as part of the callback extension.

## LlmChatCompletionSummary
A single instance of this event should be created for each chat completion request. The event will be shown in
JSON format with comments, but the actual event should be created as a New Relic custom event. A single summary even
will be created for multiple message events.

```json5
{
  "id": "lkajsdfl", // the unique id for the request. Use the completion ID from the LLM response via `kwargs`. If not found there, look in the `response_obj`. If neither are found, generate a UUID4 and log a warning.
  "trace_id": "lkasdf-9adsf", // retrieve the current trace id from the New Relic agent via `newrelic.agent.current_trace_id()` or log an warning if not set and exit out of the processing
  "span_id": "lkjasfdlkaj", // retrieve the current span id from the New Relic agent via `newrelic.agent.current_span_id()` or log an warning if not set and  exit out of the processing
  "request.model": "gpt-4-", // the model used for the request
  "response.model": "gpt-4o-2024-11-20", // the model used for the response
  "response.choices.finish_reason": "stop", // the finish reason for the response. If not found, set to "unknown"
  "response.number_of_messages": 3, // the total number of messages in the request
  "vendor": "openai", // the vendor used for the request
  "response.usage.prompt_tokens": 123, // the number of prompt tokens reported by the usage statistics from the LLM
  "response.usage.completion_tokens": 3123, // the number of completion tokens reported by the usage statistics from the LLM
  "response.usage.total_tokens": 3401 // the total number of tokens reported by the usage statistics from the LLM
}
```

The finish_reason will need to be retrieved from the LLM response. Look at other extensions to see how they retrieve
the finish reason from the callback context/object.

The `response.number_of_messages` is the total number of messages in the request. This includes assistant or tool messages.
This number will be set after the conversation has completed and is the total number of messages sent and/or received.

## LlmChatCompletionMessage
Each message in a reqeust to the LLM will be a separate message. This is normally for messages such as context, user,
and response but can include other types of messages such as assistant or tool messages. Each of the messages belong 
to a LlmChatCompletionSummary event based on `LlmChatCompletionSummary.id`. Each message will look similar to the 
following JSON format with comments, but the actual event should be created as a New Relic custom event.

Each message is the full message and will be provided by LiteLLM. Messages are available in the callback handler 
through `kwargs.get("messages", [])`. If the response contains choices, include those as if they are messages.

```json5
{
  "completion_id": "lkajsdfl", // comes from `LlmChatCompletionSummary.id`
  "trace_id": "lkasdf-9adsf", // retrieve the current trace id from the New Relic agent via `newrelic.agent.current_trace_id()` or log an warning if not set and  exit out of the processing
  "span_id": "lkjasfdlkaj", // retrieve the current span id from the New Relic agent via `newrelic.agent.current_span_id()` or log an warning if not set and  exit out of the processing
  "content": "This is my content", // the content of the message
  "role": "user", // the role of the message (user, system, assistant, tools, function, etc)
  "sequence": 1, // the sequence number of the message in the request/response
  "response.model": "gpt-4o-2024-11-20", // the model used for the response
  "vendor": "openai" // the vendor used for the request
}
```

The `sequence` number should be ordered from 0-N based on the order of _ALL_ messages were sent and/or received to the LLM.
As LiteLLM is server for OpenAI clients, the order that the messages are defined in the OpenAI request and response should
be the order of the sequence number. All request messages will be the first message(s). Then the response messages(s)
will follow. Any type of response messaage should be included as an event and have the correct sequence number based on
where it appears in the request/response.

Each message receives an incremental sequence number. As an example, if the request sends a context message (0), a user message (1), and the response
is the assistant message (2), then the sequence numbers would be 0, 1, and 2 respectively.