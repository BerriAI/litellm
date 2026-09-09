# Chat completions callback baseline

Source revision: `67d24ff9d0`

The checkout was clean before implementation. Python behavior in `litellm/utils.py`, `litellm/litellm_core_utils/litellm_logging.py`, and the Anthropic and Bedrock provider handlers is the compatibility source

## Callback contract

| Provider | Operation | Placement | Invocation | Arguments and return | Exception | Owner |
| --- | --- | --- | --- | --- | --- | --- |
| Anthropic | Async deployment pre-call | Before provider selection and request construction | Awaited inline in the caller task on the async API | Shallow wrapper argument dictionary; each non-`None` result replaces the dictionary read by the next hook | Ordinary exceptions escape | Call invocation |
| Anthropic | `Logging.pre_call` | After transformation and header construction, before body encoding and send | Direct synchronous call on sync and async APIs | Original input list, logging API key, and `additional_args`; callback return is ignored and in-place mutation remains visible | Per-callback `Exception` handling stays inside the existing dispatcher; an escaping exception prevents send | Logging object and call invocation |
| Anthropic | Deployment success | After response normalization, before public completion | Awaited inline in the caller task | Current hook argument dictionary and public response; non-`None` results replace the public response | Ordinary exceptions escape through the existing success path | Call invocation |
| Anthropic | Success and failure logging | After the deployment result according to the sync, async, internal-call, and fallback branches | Existing inline, executor, or background schedule | Existing `Logging` object and callback-visible request, response, or original exception; logging-only replacement does not replace the public response | Existing dispatcher policy | Logging task or call invocation according to schedule |
| Bedrock Converse | Async deployment pre-call | Before provider selection and request construction | Awaited inline in the caller task on the async API | Same replacement contract as Anthropic | Ordinary exceptions escape | Call invocation |
| Bedrock Converse | `Logging.pre_call` | After JSON serialization and SigV4 authorization, before provider send | Direct synchronous call on sync and async APIs | Original input list, logging API key, serialized `complete_input_dict`, and a `HeadersDict`; return is ignored | Same dispatcher policy as Anthropic | Logging object and call invocation |
| Bedrock Converse | Deployment success and terminal logging | Same lifecycle branches as Anthropic | Existing awaited, direct, executor, or background schedule | Same response replacement and logging-only replacement rules as Anthropic | Existing dispatcher policy | Call invocation or background logging owner |

## Object and read map

| Object | Anthropic | Bedrock Converse |
| --- | --- | --- |
| Boundary argument dictionary | Retained in full before deployment hooks; a returned deployment dictionary becomes the current hook dictionary without clearing the original | Same |
| Input messages | The `Logging.pre_call` `input` aliases the wrapper-selected messages object | Same |
| Callback body | A new provider-shaped dictionary whose selected optional-parameter values preserve the wrapper-created aliases; it is encoded after `pre_call` | A serialized string created from the provider body before `pre_call` |
| Execution body | Read from the retained structured callback body at send, so nested mutation reaches the wire; replacing `additional_args["complete_input_dict"]` does not redirect the retained body | Captured bytes are authorized before `pre_call`; body mutation and field replacement cannot redirect those bytes |
| Execution headers | The independently retained header object is read at send; in-place mutation reaches transport and replacing `additional_args["headers"]` does not redirect it | The independently retained `HeadersDict` is read at send while authorization remains coupled to the captured body |
| Logging object | The wrapper-selected instance is restored after deployment pre-call replacement and reused for terminal dispatch | Same |
| Response | Core normalization produces the response used to construct the existing `ModelResponse`; deployment replacement changes the public response | Same |
| Logging-only response replacement | Retained by later logging operations but does not replace the public response | Same |
| Exception | The original Python exception remains the escaping object where Python propagates it; core classification must not replace it with a string | Same |
| Release point | Request roots release after the call and any scheduled logging owner finish; cleanup drops references without clearing shared containers | Same |

Admission is the only replay boundary. A decline must happen before opening a call session, invoking a callback, reading credentials, consuming deferred input, or performing provider I/O. Every error after admission remains on the native lifecycle and cannot select legacy Python execution

## Core capability boundaries

Callback is the umbrella term. Core models each callback contract as a focused capability instead of reproducing Python's `CustomGuardrail(CustomLogger)` inheritance

| Capability | Existing Python contract | Core responsibility |
| --- | --- | --- |
| `PreCallHooks` | `async_pre_call_hook` | Apply request-level allow, replacement, or rejection before provider request preparation |
| Route preparation | No callback contract | Resolve the provider request, URL, headers, body, credentials, and authorization at the route-defined phase |
| `ModerationHooks` | `async_moderation_hook` | Apply allow, replacement, or rejection to the prepared provider request |
| Prepared-call logging | `Logging.pre_call` | Invoke direct logging with callback-visible request roots at the route-defined phase |
| Deployment hooks | `async_pre_call_deployment_hook` and post-call deployment hooks | Run once per deployment attempt and adopt replacements where Python does |
| `TerminalDispatcher` | success and failure logging handlers | Deliver the authoritative terminal record without changing the public response or original error |

Guardrails implement pre-call or moderation hook capabilities. They do not prepare provider requests, call providers, or independently dispatch terminal failures. The shared lifecycle converts their rejection into the same terminal failure path used by provider and transformation errors

`CallLifecycle` owns both typed native execution and the operation plan used by Python-backed routes. The Python loop only schedules direct operations on the caller's Python task so synchronous callbacks retain `contextvars`, event-loop identity, and nested-call behavior. Core selects every transition and whether an operation is awaited; the bridge only maps an abstract operation to its Python adapter method.

Long-lived streaming calls own a completion guard. HTTP streams, Responses WebSockets, and Realtime WebSockets dispatch a cancelled terminal record from that guard when the consumer drops the call before normal settlement.

Destination-specific callback transport belongs outside core. The LiteLLM Python proxy HTTP logger and its batching worker live in `litellm-ai-gateway`; core exposes only the destination-agnostic `CustomLogger` and terminal dispatcher contracts.

The Rust bridge exposes Bedrock callback headers through `requests.structures.CaseInsensitiveDict` instead of botocore’s concrete `HeadersDict` type. Lookup, mutation, deletion, and readback are case-insensitive; iteration uses the most recently assigned spelling of a key. Core selects header behavior independently of body serialization policy
