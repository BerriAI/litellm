This directory owns provider-independent Messages call orchestration: the entrypoint, call envelopes, provider selection, credential resolution, transport coordination, hooks, and stream lifecycle. Shared API data contracts belong in `litellm-llms-types::messages`, adapter contracts and execution inputs in `llms/src/base_llm/messages`, and provider implementations in `llms/src/<provider>/messages`

Select concrete provider adapters and invoke their contracts. Delegate authentication policy, beta selection, payload rewriting, and response interpretation to those adapters. Keep provider policy out of request preparation and transport handlers. Calling a concrete provider helper for every provider is still a policy dependency

Route types such as `MessagesCall`, prepared requests, and response wrappers containing live streams describe execution. Reuse the shared Messages payload types inside them instead of defining another request or response schema here

Preserve the order of validation, normalization, caller-requested parameter removal, and provider transformation when that order affects observable behavior. Test provider dispatch, auth precedence, header handling, transformations, and responses through behavior, not source structure
