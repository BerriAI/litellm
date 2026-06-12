# Translation v2 differential report (anthropic + bedrock + openai + google + azure + xai + the compat_sdk family: wave 1a + wave 2a)

v1 and v2 run over the same corpus; every row must be IDENTICAL (or an
explained FALLBACK that v1 serves) for a provider's flag to turn on.
Bedrock and google rows additionally pin the characterization-corpus
snapshot, so each row proves snapshot == v1-at-HEAD == v2. Regenerate with:
`python -m tests.test_litellm.translation.generate_differential_report`

- commit: c52360173d

## anthropic: request bodies (v1 map_openai_params + transform_request vs v2)

- IDENTICAL: assistant_text_and_tool_call
- IDENTICAL: cache_control_everywhere
- IDENTICAL: cache_control_on_string_message
- IDENTICAL: cache_control_on_tool_result
- IDENTICAL: current_model_default_max_tokens
- IDENTICAL: duplicate_tool_call_ids_dedupe
- IDENTICAL: empty_user_content_placeholder
- IDENTICAL: final_assistant_text_rstripped
- IDENTICAL: https_image_url
- IDENTICAL: image_data_uri
- IDENTICAL: image_format_override
- IDENTICAL: max_completion_tokens
- IDENTICAL: multiturn_stop_stream
- IDENTICAL: no_max_tokens_legacy_model
- IDENTICAL: parallel_tool_calls_false
- IDENTICAL: parallel_tool_results_merge
- IDENTICAL: parallel_with_string_none_choice
- IDENTICAL: parallel_with_tool_choice
- IDENTICAL: pydantic_style_schema_filtered
- IDENTICAL: reasoning_effort_high_no_max_tokens
- IDENTICAL: reasoning_effort_low
- IDENTICAL: reasoning_effort_none
- IDENTICAL: response_format_json_object_current
- IDENTICAL: response_format_json_object_legacy_noop
- IDENTICAL: response_format_json_schema_current
- IDENTICAL: response_format_json_schema_legacy_tool
- IDENTICAL: response_format_with_thinking_no_forced_choice
- IDENTICAL: stop_as_string
- IDENTICAL: stop_whitespace_kept_without_drop_params
- IDENTICAL: system_and_sampling
- IDENTICAL: system_as_array
- IDENTICAL: temperature_int_stays_int
- IDENTICAL: text
- IDENTICAL: thinking_explicit
- IDENTICAL: thinking_history_blocks
- IDENTICAL: thinking_no_max_tokens_bumps
- IDENTICAL: tool_call_roundtrip
- IDENTICAL: tool_choice_dict_forms
- IDENTICAL: tool_choice_none
- IDENTICAL: tool_choice_required
- IDENTICAL: tool_choice_specific
- IDENTICAL: tool_name_sanitization_with_history
- IDENTICAL: tool_result_parts_not_placeholdered
- IDENTICAL: tool_schema_missing_parameters
- IDENTICAL: tool_schema_type_coerced
- IDENTICAL: tool_use_id_sanitized
- IDENTICAL: tool_without_description
- IDENTICAL: tools_auto
- IDENTICAL: top_k
- IDENTICAL: user_email_skipped
- IDENTICAL: user_metadata
- IDENTICAL: whitespace_text_part_placeholder

## anthropic: responses (v1 transform_response vs v2)

- IDENTICAL: json_tool
- IDENTICAL: text
- IDENTICAL: thinking
- IDENTICAL: tools

## anthropic: streams (v1 CustomStreamWrapper replay vs v2 engine/stream)

- IDENTICAL: text
- IDENTICAL: thinking
- IDENTICAL: tools

## openai_compat: request bodies (v1 map_openai_params + transform_request vs v2)

- IDENTICAL: cache_control_stripped_everywhere
- IDENTICAL: image_base64
- IDENTICAL: image_url_string_to_object
- IDENTICAL: max_completion_tokens
- IDENTICAL: multiturn_stop_list_stream
- IDENTICAL: parallel_tool_calls_false
- IDENTICAL: response_format_json_object
- IDENTICAL: response_format_json_schema_strict
- IDENTICAL: system_and_sampling
- IDENTICAL: temperature_int_stays_int
- IDENTICAL: text
- IDENTICAL: tool_call_compact_arguments_roundtrip
- IDENTICAL: tool_call_odd_spacing_and_blank_arguments_roundtrip
- IDENTICAL: tool_call_roundtrip
- IDENTICAL: tool_choice_none
- IDENTICAL: tool_choice_required
- IDENTICAL: tool_choice_specific
- IDENTICAL: tools_auto
- IDENTICAL: tools_strict
- FALLBACK (v1 serves it): both_max_tokens_keys (both max_tokens and max_completion_tokens)
- FALLBACK (v1 serves it): consecutive_user_messages (consecutive user messages)
- FALLBACK (v1 serves it): empty_tools_list (empty tools list)
- FALLBACK (v1 serves it): gpt5_model (OpenAIGPT5Config)
- FALLBACK (v1 serves it): http_pdf_file_id (messages)
- FALLBACK (v1 serves it): image_detail_key (image_url detail/format)
- FALLBACK (v1 serves it): legacy_function_call (function_call)
- FALLBACK (v1 serves it): message_name_field (message name field)
- FALLBACK (v1 serves it): o_series_model (OpenAIOSeriesConfig)
- FALLBACK (v1 serves it): reasoning_effort_plain_gpt (reasoning_effort)
- FALLBACK (v1 serves it): response_format_on_gpt4 (outside v1's supported set)
- FALLBACK (v1 serves it): single_text_content_list (single-text content list)
- FALLBACK (v1 serves it): stop_string_form (string-form stop)
- FALLBACK (v1 serves it): stream_options_unsupported (stream_options)
- FALLBACK (v1 serves it): top_k_not_openai (top_k)
- FALLBACK (v1 serves it): user_param_model_list_gate (open_ai_chat_completion_models)

## openai_compat: responses (v1 convert_to_model_response_object vs v2)

- IDENTICAL: cached_and_reasoning_usage_details
- IDENTICAL: compat_finish_reason_mapped
- IDENTICAL: compat_finish_reason_unmapped
- IDENTICAL: reasoning_content_key
- IDENTICAL: text
- IDENTICAL: think_tag_extraction
- IDENTICAL: tool_calls_rewrites_stop
- IDENTICAL: text (pre-set model_response.model='someprovider/pre-call-model'; the compat provider/wire-model re-prefix arm)
- IDENTICAL: text (pre-set model_response.model='no-slash-model'; the compat provider/wire-model re-prefix arm)

## openai_compat: streams (v1 CustomStreamWrapper over SDK chunks vs v2 fold)

- IDENTICAL: empty_keepalive_swallowed
- IDENTICAL: service_tier_wire_carried
- IDENTICAL: text
- IDENTICAL: text_no_leading_role
- IDENTICAL: tools
- SEAM CONTRACT: usage tail (v2 passes the wire choices=[] usage chunk through; v1's wrapper synthesizes its final usage chunk from it, which is the streaming seam's envelope to reproduce)

## xai: request bodies (characterization snapshot == v1-at-HEAD == v2, canonical JSON; v1 = get_optional_params('xai') + transform_request)

- IDENTICAL: cache_control_stripped_grok4
- IDENTICAL: image_base64_grok4
- IDENTICAL: image_url_string_grok4
- IDENTICAL: max_tokens_grok3mini
- IDENTICAL: multiturn_stop_list_stream_grok3
- IDENTICAL: nonuser_name_stripped_grok4
- IDENTICAL: parallel_tool_calls_false_grok4
- IDENTICAL: reasoning_effort_grok3mini
- IDENTICAL: reasoning_effort_grok_code_fast
- IDENTICAL: response_format_json_object_grok4
- IDENTICAL: response_format_json_schema_strict_kept_grok4
- IDENTICAL: stop_list_grok2
- IDENTICAL: system_and_sampling_grok4
- IDENTICAL: temperature_int_stays_int_grok4
- IDENTICAL: text_grok4
- IDENTICAL: tool_call_roundtrip_compact_grok4
- IDENTICAL: tool_choice_required_grok4
- IDENTICAL: tool_choice_specific_grok4
- IDENTICAL: tools_auto_grok4
- IDENTICAL: tools_strict_stripped_grok4
- IDENTICAL: user_param_grok4
- FALLBACK (v1 raises UnsupportedParamsError): frequency_penalty_on_grok4 (frequency_penalty)
- FALLBACK (v1 raises UnsupportedParamsError): max_completion_tokens_any_grok (max_completion_tokens)
- FALLBACK (v1 raises UnsupportedParamsError): max_completion_tokens_grok3mini (max_completion_tokens)
- FALLBACK (v1 raises UnsupportedParamsError): reasoning_effort_on_non_reasoning_grok4 (reasoning_effort on non-reasoning xai model)
- FALLBACK (v1 raises UnsupportedParamsError): stop_on_grok3mini (stop on grok-3-mini)
- FALLBACK (v1 raises UnsupportedParamsError): stop_on_grok4 (stop on grok-4-0709)
- FALLBACK (v1 raises UnsupportedParamsError): stop_on_grok_code_fast (stop on grok-code-fast-1)
- FALLBACK (v1 serves it): both_max_tokens_keys (both max_tokens and max_completion_tokens)
- FALLBACK (v1 serves it): consecutive_user_messages (consecutive user messages)
- FALLBACK (v1 serves it): empty_tools_list (empty tools list)
- FALLBACK (v1 serves it): explicit_stream_false_reaches_wire (explicit stream: false)
- FALLBACK (v1 serves it): frequency_penalty_supported_family_outside_ir (frequency_penalty)
- FALLBACK (v1 serves it): image_detail_key (image_url detail/format)
- FALLBACK (v1 serves it): logprobs_outside_ir (logprobs)
- FALLBACK (v1 serves it): nested_tool_strict_below_function (nested 'strict' key)
- FALLBACK (v1 serves it): presence_penalty_outside_ir (presence_penalty)
- FALLBACK (v1 serves it): seed_outside_ir (seed)
- FALLBACK (v1 serves it): stream_options_outside_ir (stream_options)
- FALLBACK (v1 serves it): string_form_stop_supported_family (string-form stop)
- FALLBACK (v1 serves it): top_k_not_an_xai_param (top_k)
- FALLBACK (v1 serves it): use_xai_oauth_pkce_flow (PKCE)
- FALLBACK (v1 serves it): user_message_name_forwarded_by_v1 (message name field)
- FALLBACK (v1 serves it): web_search_options_responses_bridge (Responses-API bridge)

## xai: responses (snapshot == v1 XAIChatConfig.transform_response == v2; the LIVE httpx-path normalizer incl. the usage post-steps)

- IDENTICAL: cached_tokens_usage_passthrough
- IDENTICAL: finish_empty_string_with_tool_calls
- IDENTICAL: finish_stop_with_tool_calls_rewrites
- IDENTICAL: numeric_string_usage_coerced
- IDENTICAL: reasoning_tokens_already_folded_idempotent
- IDENTICAL: reasoning_tokens_folded
- IDENTICAL: text_basic
- IDENTICAL: total_tokens_normalized_up
- IDENTICAL: websearch_sources_and_citations

## xai: streams (snapshot == v1 line-seam replay through XAIChatCompletionStreamingHandler + CustomStreamWrapper('xai') == v2 xai dialect)

- IDENTICAL: citations_dropped_by_the_dict_path
- IDENTICAL: empty_keepalive_swallowed
- IDENTICAL: reasoning_content
- IDENTICAL: reasoning_renamed
- IDENTICAL: refusal_rides_content_deltas
- IDENTICAL: text
- IDENTICAL: text_no_leading_role
- IDENTICAL: tools_typeless_continuation
- SEAM CONTRACT: usage_tail_include_usage (v1's chunk_parser injects a dummy choice so the wrapper swallows the tail and synthesizes the final usage chunk; v2 passes the wire choices=[] chunk through with the FOLDED usage for the streaming seam to synthesize from)
- IDENTICAL: usage_withheld_on_content_chunks

## cerebras: request bodies (v1 get_optional_params('cerebras') + transform_request vs v2 compat_sdk)

- IDENTICAL: max_completion_tokens
- IDENTICAL: response_format_json_object
- IDENTICAL: response_format_json_schema_strict
- IDENTICAL: stop_list
- IDENTICAL: stream_true
- IDENTICAL: system_and_sampling
- IDENTICAL: temperature_int_stays_int
- IDENTICAL: text
- IDENTICAL: tool_call_compact_roundtrip
- IDENTICAL: tool_choice_specific
- IDENTICAL: tools_auto
- IDENTICAL: user_param
- FALLBACK (v1 raises UnsupportedParamsError): cerebras:parallel_tool_calls (parallel_tool_calls)
- FALLBACK (v1 raises UnsupportedParamsError): cerebras:reasoning_effort_non_reasoning_model (reasoning_effort on non-reasoning cerebras model)
- FALLBACK (v1 serves it): cerebras:both_max_tokens_keys (both max_tokens and max_completion_tokens)
- FALLBACK (v1 serves it): cerebras:explicit_stream_false (explicit stream: false)
- FALLBACK (v1 serves it): cerebras:message_name_field (message name field)
- FALLBACK (v1 serves it): cerebras:seed_outside_ir (seed)
- FALLBACK (v1 serves it): cerebras:string_form_stop (string-form stop)

## cometapi: request bodies (v1 get_optional_params('cometapi') + transform_request vs v2 compat_sdk)

- IDENTICAL: max_completion_tokens
- IDENTICAL: parallel_tool_calls_false
- IDENTICAL: response_format_json_object
- IDENTICAL: response_format_json_schema_strict
- IDENTICAL: stop_list
- IDENTICAL: stream_true
- IDENTICAL: system_and_sampling
- IDENTICAL: temperature_int_stays_int
- IDENTICAL: text
- IDENTICAL: tool_call_compact_roundtrip
- IDENTICAL: tool_choice_specific
- IDENTICAL: tools_auto
- FALLBACK (v1 raises UnsupportedParamsError): cometapi:reasoning_effort (reasoning_effort)
- FALLBACK (v1 serves it): cometapi:both_max_tokens_keys (both max_tokens and max_completion_tokens)
- FALLBACK (v1 serves it): cometapi:explicit_stream_false (explicit stream: false)
- FALLBACK (v1 serves it): cometapi:message_name_field (message name field)
- FALLBACK (v1 serves it): cometapi:seed_outside_ir (seed)
- FALLBACK (v1 serves it): cometapi:string_form_stop (string-form stop)
- FALLBACK (v1 serves it): cometapi:top_k_extra_body (extra_body)
- FALLBACK (v1 serves it): cometapi:user_model_list_gate (user)

## deepinfra: request bodies (v1 get_optional_params('deepinfra') + transform_request vs v2 compat_sdk)

- IDENTICAL: max_completion_tokens
- IDENTICAL: response_format_json_object
- IDENTICAL: response_format_json_schema_strict
- IDENTICAL: stop_list
- IDENTICAL: stream_true
- IDENTICAL: system_and_sampling
- IDENTICAL: temperature_int_stays_int
- IDENTICAL: text
- IDENTICAL: tool_call_compact_roundtrip
- IDENTICAL: tools_auto
- FALLBACK (v1 raises UnsupportedParamsError): deepinfra:parallel_tool_calls (parallel_tool_calls)
- FALLBACK (v1 raises UnsupportedParamsError): deepinfra:reasoning_effort (reasoning_effort)
- FALLBACK (v1 raises UnsupportedParamsError): deepinfra:tool_choice_required (tool_choice)
- FALLBACK (v1 raises UnsupportedParamsError): deepinfra:tool_choice_specific (tool_choice)
- FALLBACK (v1 serves it): deepinfra:both_max_tokens_keys (both max_tokens and max_completion_tokens)
- FALLBACK (v1 serves it): deepinfra:explicit_stream_false (explicit stream: false)
- FALLBACK (v1 serves it): deepinfra:message_name_field (message name field)
- FALLBACK (v1 serves it): deepinfra:seed_outside_ir (seed)
- FALLBACK (v1 serves it): deepinfra:string_form_stop (string-form stop)
- FALLBACK (v1 serves it): deepinfra:tool_message_list_content (list-form tool content)
- FALLBACK (v1 serves it): deepinfra:top_k_extra_body (extra_body)
- FALLBACK (v1 serves it): deepinfra:user_model_list_gate (user)

## featherless_ai: request bodies (v1 get_optional_params('featherless_ai') + transform_request vs v2 compat_sdk)

- IDENTICAL: max_completion_tokens
- IDENTICAL: stop_list
- IDENTICAL: stream_true
- IDENTICAL: system_and_sampling
- IDENTICAL: temperature_int_stays_int
- IDENTICAL: text
- FALLBACK (v1 raises UnsupportedParamsError): featherless_ai:parallel_tool_calls (parallel_tool_calls)
- FALLBACK (v1 raises UnsupportedParamsError): featherless_ai:reasoning_effort (reasoning_effort)
- FALLBACK (v1 raises UnsupportedParamsError): featherless_ai:response_format (response_format)
- FALLBACK (v1 raises UnsupportedParamsError): featherless_ai:tools (tools)
- FALLBACK (v1 serves it): featherless_ai:both_max_tokens_keys (both max_tokens and max_completion_tokens)
- FALLBACK (v1 serves it): featherless_ai:explicit_stream_false (explicit stream: false)
- FALLBACK (v1 serves it): featherless_ai:message_name_field (message name field)
- FALLBACK (v1 serves it): featherless_ai:seed_outside_ir (seed)
- FALLBACK (v1 serves it): featherless_ai:string_form_stop (string-form stop)
- FALLBACK (v1 serves it): featherless_ai:user_model_list_gate (user)

## hyperbolic: request bodies (v1 get_optional_params('hyperbolic') + transform_request vs v2 compat_sdk)

- IDENTICAL: response_format_json_object
- IDENTICAL: response_format_json_schema_strict
- IDENTICAL: stop_list
- IDENTICAL: stream_true
- IDENTICAL: system_and_sampling
- IDENTICAL: temperature_int_stays_int
- IDENTICAL: text
- IDENTICAL: tool_call_compact_roundtrip
- IDENTICAL: tool_choice_specific
- IDENTICAL: tools_auto
- IDENTICAL: user_param
- FALLBACK (v1 raises UnsupportedParamsError): hyperbolic:max_completion_tokens (max_completion_tokens)
- FALLBACK (v1 raises UnsupportedParamsError): hyperbolic:parallel_tool_calls (parallel_tool_calls)
- FALLBACK (v1 raises UnsupportedParamsError): hyperbolic:reasoning_effort (reasoning_effort)
- FALLBACK (v1 serves it): hyperbolic:both_max_tokens_keys (both max_tokens and max_completion_tokens)
- FALLBACK (v1 serves it): hyperbolic:explicit_stream_false (explicit stream: false)
- FALLBACK (v1 serves it): hyperbolic:message_name_field (message name field)
- FALLBACK (v1 serves it): hyperbolic:seed_outside_ir (seed)
- FALLBACK (v1 serves it): hyperbolic:string_form_stop (string-form stop)

## lambda_ai: request bodies (v1 get_optional_params('lambda_ai') + transform_request vs v2 compat_sdk)

- IDENTICAL: max_completion_tokens
- IDENTICAL: parallel_tool_calls_false
- IDENTICAL: response_format_json_object
- IDENTICAL: response_format_json_schema_strict
- IDENTICAL: stop_list
- IDENTICAL: stream_true
- IDENTICAL: system_and_sampling
- IDENTICAL: temperature_int_stays_int
- IDENTICAL: text
- IDENTICAL: tool_call_compact_roundtrip
- IDENTICAL: tool_choice_specific
- IDENTICAL: tools_auto
- FALLBACK (v1 raises UnsupportedParamsError): lambda_ai:reasoning_effort (reasoning_effort)
- FALLBACK (v1 serves it): lambda_ai:both_max_tokens_keys (both max_tokens and max_completion_tokens)
- FALLBACK (v1 serves it): lambda_ai:explicit_stream_false (explicit stream: false)
- FALLBACK (v1 serves it): lambda_ai:message_name_field (message name field)
- FALLBACK (v1 serves it): lambda_ai:seed_outside_ir (seed)
- FALLBACK (v1 serves it): lambda_ai:string_form_stop (string-form stop)
- FALLBACK (v1 serves it): lambda_ai:user_model_list_gate (user)

## llamafile: request bodies (v1 get_optional_params('llamafile') + transform_request vs v2 compat_sdk)

- IDENTICAL: max_completion_tokens
- IDENTICAL: parallel_tool_calls_false
- IDENTICAL: response_format_json_object
- IDENTICAL: response_format_json_schema_strict
- IDENTICAL: stop_list
- IDENTICAL: stream_true
- IDENTICAL: system_and_sampling
- IDENTICAL: temperature_int_stays_int
- IDENTICAL: text
- IDENTICAL: tool_call_compact_roundtrip
- IDENTICAL: tool_choice_specific
- IDENTICAL: tools_auto
- FALLBACK (v1 raises UnsupportedParamsError): llamafile:reasoning_effort (reasoning_effort)
- FALLBACK (v1 serves it): llamafile:both_max_tokens_keys (both max_tokens and max_completion_tokens)
- FALLBACK (v1 serves it): llamafile:explicit_stream_false (explicit stream: false)
- FALLBACK (v1 serves it): llamafile:message_name_field (message name field)
- FALLBACK (v1 serves it): llamafile:seed_outside_ir (seed)
- FALLBACK (v1 serves it): llamafile:string_form_stop (string-form stop)
- FALLBACK (v1 serves it): llamafile:user_model_list_gate (user)

## lm_studio: request bodies (v1 get_optional_params('lm_studio') + transform_request vs v2 compat_sdk)

- IDENTICAL: max_completion_tokens
- IDENTICAL: parallel_tool_calls_false
- IDENTICAL: response_format_json_object
- IDENTICAL: response_format_json_schema_strict
- IDENTICAL: stop_list
- IDENTICAL: stream_true
- IDENTICAL: system_and_sampling
- IDENTICAL: temperature_int_stays_int
- IDENTICAL: text
- IDENTICAL: tool_call_compact_roundtrip
- IDENTICAL: tool_choice_specific
- IDENTICAL: tools_auto
- FALLBACK (v1 raises UnsupportedParamsError): lm_studio:reasoning_effort (reasoning_effort)
- FALLBACK (v1 raises UnsupportedParamsError): lm_studio:response_format_on_gpt4_name (outside v1's supported set)
- FALLBACK (v1 serves it): lm_studio:bare_schema_response_format (response_format)
- FALLBACK (v1 serves it): lm_studio:both_max_tokens_keys (both max_tokens and max_completion_tokens)
- FALLBACK (v1 serves it): lm_studio:explicit_stream_false (explicit stream: false)
- FALLBACK (v1 serves it): lm_studio:message_name_field (message name field)
- FALLBACK (v1 serves it): lm_studio:seed_outside_ir (seed)
- FALLBACK (v1 serves it): lm_studio:string_form_stop (string-form stop)
- FALLBACK (v1 serves it): lm_studio:user_model_list_gate (user)

## moonshot: request bodies (v1 get_optional_params('moonshot') + transform_request vs v2 compat_sdk)

- IDENTICAL: max_completion_tokens
- IDENTICAL: parallel_tool_calls_false
- IDENTICAL: response_format_json_object
- IDENTICAL: response_format_json_schema_strict
- IDENTICAL: stop_list
- IDENTICAL: stream_true
- IDENTICAL: system_and_sampling
- IDENTICAL: temperature_int_stays_int
- IDENTICAL: text
- IDENTICAL: tool_call_compact_roundtrip
- IDENTICAL: tool_choice_specific
- IDENTICAL: tools_auto
- FALLBACK (v1 raises UnsupportedParamsError): moonshot:reasoning_effort (reasoning_effort)
- FALLBACK (v1 raises UnsupportedParamsError): moonshot:tool_choice_on_kimi_thinking_preview (kimi-thinking-preview)
- FALLBACK (v1 raises UnsupportedParamsError): moonshot:tools_on_kimi_thinking_preview (kimi-thinking-preview)
- FALLBACK (v1 serves it): moonshot:both_max_tokens_keys (both max_tokens and max_completion_tokens)
- FALLBACK (v1 serves it): moonshot:explicit_stream_false (explicit stream: false)
- FALLBACK (v1 serves it): moonshot:message_name_field (message name field)
- FALLBACK (v1 serves it): moonshot:reasoning_model_tool_history (fill_reasoning_content)
- FALLBACK (v1 serves it): moonshot:seed_outside_ir (seed)
- FALLBACK (v1 serves it): moonshot:string_form_stop (string-form stop)
- FALLBACK (v1 serves it): moonshot:tool_choice_required (synthetic user message)
- FALLBACK (v1 serves it): moonshot:top_k_extra_body (extra_body)
- FALLBACK (v1 serves it): moonshot:user_model_list_gate (user)

## nebius: request bodies (v1 get_optional_params('nebius') + transform_request vs v2 compat_sdk)

- IDENTICAL: max_completion_tokens
- IDENTICAL: parallel_tool_calls_false
- IDENTICAL: response_format_json_object
- IDENTICAL: response_format_json_schema_strict
- IDENTICAL: stop_list
- IDENTICAL: stream_true
- IDENTICAL: system_and_sampling
- IDENTICAL: temperature_int_stays_int
- IDENTICAL: text
- IDENTICAL: tool_call_compact_roundtrip
- IDENTICAL: tool_choice_specific
- IDENTICAL: tools_auto
- FALLBACK (v1 raises UnsupportedParamsError): nebius:reasoning_effort (reasoning_effort)
- FALLBACK (v1 serves it): nebius:both_max_tokens_keys (both max_tokens and max_completion_tokens)
- FALLBACK (v1 serves it): nebius:explicit_stream_false (explicit stream: false)
- FALLBACK (v1 serves it): nebius:message_name_field (message name field)
- FALLBACK (v1 serves it): nebius:seed_outside_ir (seed)
- FALLBACK (v1 serves it): nebius:string_form_stop (string-form stop)
- FALLBACK (v1 serves it): nebius:user_model_list_gate (user)

## novita: request bodies (v1 get_optional_params('novita') + transform_request vs v2 compat_sdk)

- IDENTICAL: max_completion_tokens
- IDENTICAL: parallel_tool_calls_false
- IDENTICAL: response_format_json_object
- IDENTICAL: response_format_json_schema_strict
- IDENTICAL: stop_list
- IDENTICAL: stream_true
- IDENTICAL: system_and_sampling
- IDENTICAL: temperature_int_stays_int
- IDENTICAL: text
- IDENTICAL: tool_call_compact_roundtrip
- IDENTICAL: tool_choice_specific
- IDENTICAL: tools_auto
- FALLBACK (v1 raises UnsupportedParamsError): novita:reasoning_effort (reasoning_effort)
- FALLBACK (v1 serves it): novita:both_max_tokens_keys (both max_tokens and max_completion_tokens)
- FALLBACK (v1 serves it): novita:explicit_stream_false (explicit stream: false)
- FALLBACK (v1 serves it): novita:message_name_field (message name field)
- FALLBACK (v1 serves it): novita:seed_outside_ir (seed)
- FALLBACK (v1 serves it): novita:string_form_stop (string-form stop)
- FALLBACK (v1 serves it): novita:user_model_list_gate (user)

## nscale: request bodies (v1 get_optional_params('nscale') + transform_request vs v2 compat_sdk)

- IDENTICAL: response_format_json_object
- IDENTICAL: response_format_json_schema_strict
- IDENTICAL: stop_list
- IDENTICAL: stream_true
- IDENTICAL: system_and_sampling
- IDENTICAL: temperature_int_stays_int
- IDENTICAL: text
- FALLBACK (v1 raises UnsupportedParamsError): nscale:max_completion_tokens (max_completion_tokens)
- FALLBACK (v1 raises UnsupportedParamsError): nscale:parallel_tool_calls (parallel_tool_calls)
- FALLBACK (v1 raises UnsupportedParamsError): nscale:reasoning_effort (reasoning_effort)
- FALLBACK (v1 raises UnsupportedParamsError): nscale:tools (tools)
- FALLBACK (v1 serves it): nscale:both_max_tokens_keys (both max_tokens and max_completion_tokens)
- FALLBACK (v1 serves it): nscale:explicit_stream_false (explicit stream: false)
- FALLBACK (v1 serves it): nscale:message_name_field (message name field)
- FALLBACK (v1 serves it): nscale:seed_outside_ir (seed)
- FALLBACK (v1 serves it): nscale:string_form_stop (string-form stop)
- FALLBACK (v1 serves it): nscale:user_model_list_gate (user)

## nvidia_nim: request bodies (v1 get_optional_params('nvidia_nim') + transform_request vs v2 compat_sdk)

- IDENTICAL: max_completion_tokens
- IDENTICAL: parallel_tool_calls_false
- IDENTICAL: response_format_json_object
- IDENTICAL: response_format_json_schema_strict
- IDENTICAL: stop_list
- IDENTICAL: stream_true
- IDENTICAL: system_and_sampling
- IDENTICAL: temperature_int_stays_int
- IDENTICAL: text
- IDENTICAL: tool_call_compact_roundtrip
- IDENTICAL: tool_choice_specific
- IDENTICAL: tools_auto
- FALLBACK (v1 raises UnsupportedParamsError): nvidia_nim:max_completion_tokens_on_gemma (max_completion_tokens)
- FALLBACK (v1 raises UnsupportedParamsError): nvidia_nim:reasoning_effort (reasoning_effort)
- FALLBACK (v1 raises UnsupportedParamsError): nvidia_nim:stop_on_nemotron_instruct (stop)
- FALLBACK (v1 raises UnsupportedParamsError): nvidia_nim:temperature_on_nemotron_reward (temperature)
- FALLBACK (v1 raises UnsupportedParamsError): nvidia_nim:tools_on_gemma (tools)
- FALLBACK (v1 serves it): nvidia_nim:both_max_tokens_keys (both max_tokens and max_completion_tokens)
- FALLBACK (v1 serves it): nvidia_nim:explicit_stream_false (explicit stream: false)
- FALLBACK (v1 serves it): nvidia_nim:message_name_field (message name field)
- FALLBACK (v1 serves it): nvidia_nim:seed_outside_ir (seed)
- FALLBACK (v1 serves it): nvidia_nim:string_form_stop (string-form stop)
- FALLBACK (v1 serves it): nvidia_nim:user_model_list_gate (user)

## perplexity: request bodies (v1 get_optional_params('perplexity') + transform_request vs v2 compat_sdk)

- IDENTICAL: max_completion_tokens
- IDENTICAL: response_format_json_object
- IDENTICAL: response_format_json_schema_strict
- IDENTICAL: stream_true
- IDENTICAL: system_and_sampling
- IDENTICAL: temperature_int_stays_int
- IDENTICAL: text
- FALLBACK (v1 raises UnsupportedParamsError): perplexity:parallel_tool_calls (parallel_tool_calls)
- FALLBACK (v1 raises UnsupportedParamsError): perplexity:reasoning_effort (reasoning_effort)
- FALLBACK (v1 raises UnsupportedParamsError): perplexity:stop (stop)
- FALLBACK (v1 raises UnsupportedParamsError): perplexity:tool_choice (tool_choice)
- FALLBACK (v1 raises UnsupportedParamsError): perplexity:tools (tools)
- FALLBACK (v1 serves it): perplexity:both_max_tokens_keys (both max_tokens and max_completion_tokens)
- FALLBACK (v1 serves it): perplexity:explicit_stream_false (explicit stream: false)
- FALLBACK (v1 serves it): perplexity:message_name_field (message name field)
- FALLBACK (v1 serves it): perplexity:seed_outside_ir (seed)
- FALLBACK (v1 serves it): perplexity:string_form_stop (string-form stop)
- FALLBACK (v1 serves it): perplexity:top_k_extra_body (extra_body)
- FALLBACK (v1 serves it): perplexity:user_model_list_gate (user)
- FALLBACK (v1 serves it): perplexity:web_search_options (web_search_options)

## sambanova: request bodies (v1 get_optional_params('sambanova') + transform_request vs v2 compat_sdk)

- IDENTICAL: max_completion_tokens
- IDENTICAL: parallel_tool_calls_false
- IDENTICAL: response_format_json_object
- IDENTICAL: response_format_json_schema_strict
- IDENTICAL: stop_list
- IDENTICAL: stream_true
- IDENTICAL: system_and_sampling
- IDENTICAL: temperature_int_stays_int
- IDENTICAL: text
- IDENTICAL: tool_call_compact_roundtrip
- IDENTICAL: tool_choice_specific
- IDENTICAL: tools_auto
- FALLBACK (v1 raises UnsupportedParamsError): sambanova:parallel_on_non_fc_model (parallel_tool_calls)
- FALLBACK (v1 raises UnsupportedParamsError): sambanova:reasoning_effort (reasoning_effort)
- FALLBACK (v1 raises UnsupportedParamsError): sambanova:tool_choice_on_non_fc_model (tool_choice)
- FALLBACK (v1 raises UnsupportedParamsError): sambanova:tools_on_non_fc_model (tools)
- FALLBACK (v1 serves it): sambanova:both_max_tokens_keys (both max_tokens and max_completion_tokens)
- FALLBACK (v1 serves it): sambanova:explicit_stream_false (explicit stream: false)
- FALLBACK (v1 serves it): sambanova:image_content_list (non-text content block)
- FALLBACK (v1 serves it): sambanova:message_name_field (message name field)
- FALLBACK (v1 serves it): sambanova:seed_outside_ir (seed)
- FALLBACK (v1 serves it): sambanova:stream_options (stream_options)
- FALLBACK (v1 serves it): sambanova:string_form_stop (string-form stop)
- FALLBACK (v1 serves it): sambanova:top_k_extra_body (extra_body)
- FALLBACK (v1 serves it): sambanova:user_model_list_gate (user)

## together_ai: request bodies (v1 get_optional_params('together_ai') + transform_request vs v2 compat_sdk)

- IDENTICAL: max_completion_tokens
- IDENTICAL: parallel_tool_calls_false
- IDENTICAL: response_format_json_object
- IDENTICAL: response_format_json_schema_strict
- IDENTICAL: stop_list
- IDENTICAL: stream_true
- IDENTICAL: system_and_sampling
- IDENTICAL: temperature_int_stays_int
- IDENTICAL: text
- IDENTICAL: tool_call_compact_roundtrip
- IDENTICAL: tool_choice_specific
- IDENTICAL: tools_auto
- FALLBACK (v1 raises UnsupportedParamsError): together_ai:reasoning_effort (reasoning_effort)
- FALLBACK (v1 raises UnsupportedParamsError): together_ai:response_format_on_non_fc_model (response_format)
- FALLBACK (v1 raises UnsupportedParamsError): together_ai:tools_on_non_fc_model (tools)
- FALLBACK (v1 raises ValueError): together_ai:plain_text_on_gpt35_16k_name (ValueError)
- FALLBACK (v1 raises ValueError): together_ai:plain_text_on_gpt4_name (ValueError)
- FALLBACK (v1 raises ValueError): together_ai:response_format_on_gpt4_name (ValueError)
- FALLBACK (v1 raises ValueError): together_ai:temperature_only_on_gpt4_name (ValueError)
- FALLBACK (v1 serves it): together_ai:both_max_tokens_keys (both max_tokens and max_completion_tokens)
- FALLBACK (v1 serves it): together_ai:explicit_stream_false (explicit stream: false)
- FALLBACK (v1 serves it): together_ai:message_name_field (message name field)
- FALLBACK (v1 serves it): together_ai:seed_outside_ir (seed)
- FALLBACK (v1 serves it): together_ai:string_form_stop (string-form stop)
- FALLBACK (v1 serves it): together_ai:user_model_list_gate (user)

## volcengine: request bodies (v1 get_optional_params('volcengine') + transform_request vs v2 compat_sdk)

- IDENTICAL: max_completion_tokens
- IDENTICAL: stop_list
- IDENTICAL: stream_true
- IDENTICAL: system_and_sampling
- IDENTICAL: temperature_int_stays_int
- IDENTICAL: text
- IDENTICAL: tool_call_compact_roundtrip
- IDENTICAL: tool_choice_specific
- IDENTICAL: tools_auto
- FALLBACK (v1 raises UnsupportedParamsError): volcengine:parallel_tool_calls (parallel_tool_calls)
- FALLBACK (v1 raises UnsupportedParamsError): volcengine:reasoning_effort (reasoning_effort)
- FALLBACK (v1 raises UnsupportedParamsError): volcengine:response_format (response_format)
- FALLBACK (v1 serves it): volcengine:both_max_tokens_keys (both max_tokens and max_completion_tokens)
- FALLBACK (v1 serves it): volcengine:explicit_stream_false (explicit stream: false)
- FALLBACK (v1 serves it): volcengine:message_name_field (message name field)
- FALLBACK (v1 serves it): volcengine:seed_outside_ir (seed)
- FALLBACK (v1 serves it): volcengine:string_form_stop (string-form stop)
- FALLBACK (v1 serves it): volcengine:thinking_extra_body_packing (extra_body)
- FALLBACK (v1 serves it): volcengine:user_model_list_gate (user)

## wandb: request bodies (v1 get_optional_params('wandb') + transform_request vs v2 compat_sdk)

- IDENTICAL: max_completion_tokens
- IDENTICAL: parallel_tool_calls_false
- IDENTICAL: response_format_json_object
- IDENTICAL: response_format_json_schema_strict
- IDENTICAL: stop_list
- IDENTICAL: stream_true
- IDENTICAL: system_and_sampling
- IDENTICAL: temperature_int_stays_int
- IDENTICAL: text
- IDENTICAL: tool_call_compact_roundtrip
- IDENTICAL: tool_choice_specific
- IDENTICAL: tools_auto
- FALLBACK (v1 raises UnsupportedParamsError): wandb:reasoning_effort (reasoning_effort)
- FALLBACK (v1 serves it): wandb:both_max_tokens_keys (both max_tokens and max_completion_tokens)
- FALLBACK (v1 serves it): wandb:explicit_stream_false (explicit stream: false)
- FALLBACK (v1 serves it): wandb:message_name_field (message name field)
- FALLBACK (v1 serves it): wandb:seed_outside_ir (seed)
- FALLBACK (v1 serves it): wandb:string_form_stop (string-form stop)
- FALLBACK (v1 serves it): wandb:user_model_list_gate (user)

## compat_sdk family: responses (v1 convert_to_model_response_object with the SDK-path {provider}/{model} preset vs v2 + seam re-prefix arm; SDK-path members only — cometapi's no-prefix rows are below)

- IDENTICAL: cerebras cached_and_reasoning_usage_details
- IDENTICAL: cerebras text
- IDENTICAL: cerebras tool_calls_rewrites_stop
- IDENTICAL: deepinfra cached_and_reasoning_usage_details
- IDENTICAL: deepinfra text
- IDENTICAL: deepinfra tool_calls_rewrites_stop
- IDENTICAL: featherless_ai cached_and_reasoning_usage_details
- IDENTICAL: featherless_ai text
- IDENTICAL: featherless_ai tool_calls_rewrites_stop
- IDENTICAL: hyperbolic cached_and_reasoning_usage_details
- IDENTICAL: hyperbolic text
- IDENTICAL: hyperbolic tool_calls_rewrites_stop
- IDENTICAL: lambda_ai cached_and_reasoning_usage_details
- IDENTICAL: lambda_ai text
- IDENTICAL: lambda_ai tool_calls_rewrites_stop
- IDENTICAL: llamafile cached_and_reasoning_usage_details
- IDENTICAL: llamafile text
- IDENTICAL: llamafile tool_calls_rewrites_stop
- IDENTICAL: lm_studio cached_and_reasoning_usage_details
- IDENTICAL: lm_studio text
- IDENTICAL: lm_studio tool_calls_rewrites_stop
- IDENTICAL: moonshot cached_and_reasoning_usage_details
- IDENTICAL: moonshot text
- IDENTICAL: moonshot tool_calls_rewrites_stop
- IDENTICAL: nebius cached_and_reasoning_usage_details
- IDENTICAL: nebius text
- IDENTICAL: nebius tool_calls_rewrites_stop
- IDENTICAL: novita cached_and_reasoning_usage_details
- IDENTICAL: novita text
- IDENTICAL: novita tool_calls_rewrites_stop
- IDENTICAL: nscale cached_and_reasoning_usage_details
- IDENTICAL: nscale text
- IDENTICAL: nscale tool_calls_rewrites_stop
- IDENTICAL: nvidia_nim cached_and_reasoning_usage_details
- IDENTICAL: nvidia_nim text
- IDENTICAL: nvidia_nim tool_calls_rewrites_stop
- IDENTICAL: perplexity cached_and_reasoning_usage_details
- IDENTICAL: perplexity text
- IDENTICAL: perplexity tool_calls_rewrites_stop
- IDENTICAL: sambanova cached_and_reasoning_usage_details
- IDENTICAL: sambanova text
- IDENTICAL: sambanova tool_calls_rewrites_stop
- IDENTICAL: together_ai cached_and_reasoning_usage_details
- IDENTICAL: together_ai text
- IDENTICAL: together_ai tool_calls_rewrites_stop
- IDENTICAL: volcengine cached_and_reasoning_usage_details
- IDENTICAL: volcengine text
- IDENTICAL: volcengine tool_calls_rewrites_stop
- IDENTICAL: wandb cached_and_reasoning_usage_details
- IDENTICAL: wandb text
- IDENTICAL: wandb tool_calls_rewrites_stop
- IDENTICAL: perplexity citations dormancy (transform_response's annotation/citation-token enrichment is DEAD on the SDK path; citations/search_results survive via cdr's unknown-key mirror only)

## compat_sdk family: streams (v1 CustomStreamWrapper(provider) over SDK chunks vs v2 openai dialect; SDK-path members only)

- IDENTICAL: cerebras empty_keepalive_swallowed
- IDENTICAL: cerebras text
- IDENTICAL: cerebras tools
- IDENTICAL: deepinfra empty_keepalive_swallowed
- IDENTICAL: deepinfra text
- IDENTICAL: deepinfra tools
- IDENTICAL: featherless_ai empty_keepalive_swallowed
- IDENTICAL: featherless_ai text
- IDENTICAL: featherless_ai tools
- IDENTICAL: hyperbolic empty_keepalive_swallowed
- IDENTICAL: hyperbolic text
- IDENTICAL: hyperbolic tools
- IDENTICAL: lambda_ai empty_keepalive_swallowed
- IDENTICAL: lambda_ai text
- IDENTICAL: lambda_ai tools
- IDENTICAL: llamafile empty_keepalive_swallowed
- IDENTICAL: llamafile text
- IDENTICAL: llamafile tools
- IDENTICAL: lm_studio empty_keepalive_swallowed
- IDENTICAL: lm_studio text
- IDENTICAL: lm_studio tools
- IDENTICAL: moonshot empty_keepalive_swallowed
- IDENTICAL: moonshot text
- IDENTICAL: moonshot tools
- IDENTICAL: nebius empty_keepalive_swallowed
- IDENTICAL: nebius text
- IDENTICAL: nebius tools
- IDENTICAL: novita empty_keepalive_swallowed
- IDENTICAL: novita text
- IDENTICAL: novita tools
- IDENTICAL: nscale empty_keepalive_swallowed
- IDENTICAL: nscale text
- IDENTICAL: nscale tools
- IDENTICAL: nvidia_nim empty_keepalive_swallowed
- IDENTICAL: nvidia_nim text
- IDENTICAL: nvidia_nim tools
- IDENTICAL: perplexity empty_keepalive_swallowed
- IDENTICAL: perplexity text
- IDENTICAL: perplexity tools
- IDENTICAL: sambanova empty_keepalive_swallowed
- IDENTICAL: sambanova text
- IDENTICAL: sambanova tools
- IDENTICAL: together_ai empty_keepalive_swallowed
- IDENTICAL: together_ai text
- IDENTICAL: together_ai tools
- IDENTICAL: volcengine empty_keepalive_swallowed
- IDENTICAL: volcengine text
- IDENTICAL: volcengine tools
- IDENTICAL: wandb empty_keepalive_swallowed
- IDENTICAL: wandb text
- IDENTICAL: wandb tools
- IDENTICAL: perplexity wire-carried citations (body value survives the seam's citations preset; None preset where the wire carried none)
- SEAM CONTRACT: cerebras usage tail (v2 passes the wire choices=[] usage chunk through; the streaming seam owns v1's synthesized final chunk)
- SEAM CONTRACT: deepinfra usage tail (v2 passes the wire choices=[] usage chunk through; the streaming seam owns v1's synthesized final chunk)
- SEAM CONTRACT: featherless_ai usage tail (v2 passes the wire choices=[] usage chunk through; the streaming seam owns v1's synthesized final chunk)
- SEAM CONTRACT: hyperbolic usage tail (v2 passes the wire choices=[] usage chunk through; the streaming seam owns v1's synthesized final chunk)
- SEAM CONTRACT: lambda_ai usage tail (v2 passes the wire choices=[] usage chunk through; the streaming seam owns v1's synthesized final chunk)
- SEAM CONTRACT: llamafile usage tail (v2 passes the wire choices=[] usage chunk through; the streaming seam owns v1's synthesized final chunk)
- SEAM CONTRACT: lm_studio usage tail (v2 passes the wire choices=[] usage chunk through; the streaming seam owns v1's synthesized final chunk)
- SEAM CONTRACT: moonshot usage tail (v2 passes the wire choices=[] usage chunk through; the streaming seam owns v1's synthesized final chunk)
- SEAM CONTRACT: nebius usage tail (v2 passes the wire choices=[] usage chunk through; the streaming seam owns v1's synthesized final chunk)
- SEAM CONTRACT: novita usage tail (v2 passes the wire choices=[] usage chunk through; the streaming seam owns v1's synthesized final chunk)
- SEAM CONTRACT: nscale usage tail (v2 passes the wire choices=[] usage chunk through; the streaming seam owns v1's synthesized final chunk)
- SEAM CONTRACT: nvidia_nim usage tail (v2 passes the wire choices=[] usage chunk through; the streaming seam owns v1's synthesized final chunk)
- SEAM CONTRACT: perplexity usage tail (v2 passes the wire choices=[] usage chunk through; the streaming seam owns v1's synthesized final chunk)
- SEAM CONTRACT: sambanova usage tail (v2 passes the wire choices=[] usage chunk through; the streaming seam owns v1's synthesized final chunk)
- SEAM CONTRACT: together_ai usage tail (v2 passes the wire choices=[] usage chunk through; the streaming seam owns v1's synthesized final chunk)
- SEAM CONTRACT: volcengine usage tail (v2 passes the wire choices=[] usage chunk through; the streaming seam owns v1's synthesized final chunk)
- SEAM CONTRACT: wandb usage tail (v2 passes the wire choices=[] usage chunk through; the streaming seam owns v1's synthesized final chunk)

- DROPPED FROM WAVE 1A: baseten (streams ride the dedicated legacy handle_baseten_chunk wrapper branch, not the openai dialect; unregistered, typed v1 fallback; canary test_baseten_drop_canary pins the evidence)

## cometapi: responses (v1 CometAPIConfig.transform_response over httpx — LIVE on the dedicated elif, main.py:2547 — vs v2 shared openai parser with NO model preset; bare wire model, the xai R4 pin)

- IDENTICAL: reasoning_usage_details (no prefix)
- IDENTICAL: text (no prefix)
- IDENTICAL: tool_calls (no prefix)

## cometapi: streams (v1 line-seam replay through CometAPIChatCompletionStreamingHandler + CustomStreamWrapper('cometapi') vs v2 cometapi parser + the shared xai chunk dialect)

- IDENTICAL: extras_dropped
- IDENTICAL: native_reasoning_content
- IDENTICAL: reasoning_rename
- IDENTICAL: text
- IDENTICAL: tools
- SEAM CONTRACT: usage tail (v2 passes the wire choices=[] usage chunk through; the streaming seam owns v1's synthesized final chunk)

## azure: request bodies (v1 api-version-aware map_openai_params + transform_request vs v2)

- IDENTICAL: deployment_with_base_model
- IDENTICAL: gpt5_chat_is_plain_azure
- IDENTICAL: image_base64
- IDENTICAL: image_url_string_to_object
- IDENTICAL: max_completion_tokens
- IDENTICAL: multiturn_stop_list_stream
- IDENTICAL: parallel_tool_calls_false
- IDENTICAL: response_format_json_object
- IDENTICAL: response_format_json_schema_strict
- IDENTICAL: response_format_on_gpt4_deployment
- IDENTICAL: system_and_sampling
- IDENTICAL: temperature_int_stays_int
- IDENTICAL: text
- IDENTICAL: tool_call_roundtrip
- IDENTICAL: tool_choice_required_current_api
- IDENTICAL: tool_choice_specific
- IDENTICAL: tool_choice_unparseable_api_version_passthrough
- IDENTICAL: tools_auto
- IDENTICAL: tools_strict
- FALLBACK (v1 serves it): cache_control_in_messages (cache_control inside messages)
- FALLBACK (v1 serves it): cache_control_in_tools (cache_control inside tools)
- FALLBACK (v1 serves it): explicit_stream_false (explicit stream: false)
- FALLBACK (v1 serves it): gpt5_model (AzureOpenAIGPT5Config)
- FALLBACK (v1 serves it): gpt5_series_prefix (AzureOpenAIGPT5Config)
- FALLBACK (v1 serves it): o_series_empty_base_model_falls_to_deployment (AzureOpenAIO1Config)
- FALLBACK (v1 serves it): o_series_substring_deployment (AzureOpenAIO1Config)
- FALLBACK (v1 serves it): o_series_via_base_model (AzureOpenAIO1Config)
- FALLBACK (v1 serves it): reasoning_effort_plain_azure (reasoning_effort)
- FALLBACK (v1 serves it): response_format_gpt35 (json-tool strategy)
- FALLBACK (v1 serves it): response_format_gpt_3_5_normalized (json-tool strategy)
- FALLBACK (v1 serves it): response_format_pre_2024_08_api (response_format needs api_version)
- FALLBACK (v1 serves it): response_format_with_unwired_api_version (not wired)
- FALLBACK (v1 serves it): shared_guard_string_stop (string-form stop)
- FALLBACK (v1 serves it): tool_choice_pre_2023_12_api (tool_choice needs api_version)
- FALLBACK (v1 serves it): tool_choice_required_2024_05_api (tool_choice='required' is unsupported)
- FALLBACK (v1 serves it): tool_choice_with_unwired_api_version (not wired)
- FALLBACK (v1 serves it): user_param (user param)

## azure: request bodies (characterization snapshot == v1-at-HEAD == v2, canonical JSON)

- FALLBACK (v1 serves it): cache_control_messages (list-form system content)
- FALLBACK (v1 serves it): cache_control_tools (cache_control inside tools)
- IDENTICAL: image_base64
- IDENTICAL: image_url
- IDENTICAL: max_completion_tokens
- IDENTICAL: multi_turn
- FALLBACK (v1 serves it): params_sampling (user param)
- FALLBACK (v1 serves it): pdf_base64 (messages)
- IDENTICAL: plain_text
- IDENTICAL: response_format_json_object
- IDENTICAL: response_format_json_schema
- IDENTICAL: system_prompt
- IDENTICAL: tools_basic
- IDENTICAL: tools_forced_choice
- IDENTICAL: tools_parallel
- IDENTICAL: tools_streamed_args_roundtrip

## azure: responses (v1 convert_to_model_response_object with azure.py's args vs v2)

- IDENTICAL: content_and_prompt_filter_results
- IDENTICAL: tool_calls_rewrites_stop
- IDENTICAL: corpus text_basic
- IDENTICAL: corpus tool_calls
- IDENTICAL: azure_ai model rename (v1 preset + convert re-prefix)

## azure: streams (v1 CustomStreamWrapper('azure') over SDK chunks vs v2 azure dialect)

- IDENTICAL: model_reread_from_chunks
- IDENTICAL: text_with_filter_results
- IDENTICAL: tools_with_filter_results
- IDENTICAL: corpus text_stream
- IDENTICAL: corpus tool_stream

## azure_ai: request bodies (v1 AzureAIStudioConfig chain vs v2)

- IDENTICAL: image_content_list_not_flattened
- IDENTICAL: max_completion_tokens
- IDENTICAL: response_format_json_schema
- IDENTICAL: stream_true
- IDENTICAL: system_and_sampling
- IDENTICAL: text
- IDENTICAL: tool_call_roundtrip
- IDENTICAL: tools_without_tool_choice
- FALLBACK (v1 serves it): cache_control_forwarded (cache_control inside tools)
- FALLBACK (v1 serves it): grok_model (XAIChatConfig)
- FALLBACK (v1 serves it): o_series_name (OpenAIOSeriesConfig)
- FALLBACK (v1 serves it): text_only_content_list_flatten (flattens it to a string)
- FALLBACK (v1 serves it): tool_choice_model_map_gated (tool_choice on azure_ai is model-map gated)
- FALLBACK (v1 serves it): user_param (user param)

## azure_ai_anthropic: request bodies (v1 AzureAnthropicConfig chain vs v2, no model spoof)

- IDENTICAL: response_format_json_tool_model
- IDENTICAL: response_format_output_format_model
- IDENTICAL: system_and_sampling
- IDENTICAL: text
- IDENTICAL: thinking_enabled
- IDENTICAL: tool_roundtrip
- IDENTICAL: tools
- FALLBACK (v1 serves it): billing_header_system_block (x-anthropic-billing-header)
- FALLBACK (v1 serves it): non_claude_model (Claude models only)

## bedrock_converse: request bodies (characterization snapshot == v1-at-HEAD == v2, canonical JSON)

- IDENTICAL: cache_control_messages
- IDENTICAL: cache_control_tools
- IDENTICAL: full_combo
- IDENTICAL: image_base64
- SKIPPED (corpus): image_url (v1 downloads URL media for bedrock transforms (network))
- IDENTICAL: multi_turn
- IDENTICAL: params_sampling
- FALLBACK (v1 serves it): pdf_base64 (file/document parts are outside the v2 inbound surface)
- IDENTICAL: plain_text
- IDENTICAL: reasoning_effort_low
- IDENTICAL: response_format_json_object
- IDENTICAL: response_format_json_schema
- IDENTICAL: system_prompt
- IDENTICAL: thinking_enabled
- IDENTICAL: thinking_history_blocks
- IDENTICAL: tools_basic
- IDENTICAL: tools_forced_choice
- IDENTICAL: tools_parallel
- IDENTICAL: tools_streamed_args_roundtrip
- IDENTICAL: quirk no_max_tokens_no_default (v1 in-process)
- IDENTICAL: quirk stop_whitespace_kept (v1 in-process)
- IDENTICAL: quirk thinking_budget_clamped_to_bedrock_min (v1 in-process)
- IDENTICAL: quirk thinking_rewrites_forced_choice_to_auto (v1 in-process)
- IDENTICAL: quirk tool_name_normalized (v1 in-process)
- IDENTICAL: quirk top_k_additional_field (v1 in-process)
- IDENTICAL: quirk assistant_blank_text_dropped (v1 in-process)
- IDENTICAL: quirk json_schema_with_effort_drops_forced_choice (v1 in-process)

## bedrock_invoke: request bodies (characterization snapshot == v1-at-HEAD == v2, canonical JSON)

- IDENTICAL: cache_control_messages
- IDENTICAL: cache_control_tools
- IDENTICAL: full_combo
- IDENTICAL: image_base64
- SKIPPED (corpus): image_url (v1 downloads URL media for bedrock transforms (network))
- IDENTICAL: multi_turn
- IDENTICAL: params_sampling
- FALLBACK (v1 serves it): pdf_base64 (file/document parts are outside the v2 inbound surface)
- IDENTICAL: plain_text
- IDENTICAL: reasoning_effort_low
- IDENTICAL: response_format_json_object
- IDENTICAL: response_format_json_schema
- IDENTICAL: system_prompt
- IDENTICAL: thinking_enabled
- IDENTICAL: thinking_history_blocks
- IDENTICAL: tools_basic
- IDENTICAL: tools_forced_choice
- IDENTICAL: tools_parallel
- IDENTICAL: tools_streamed_args_roundtrip
- IDENTICAL: quirk no_max_tokens_no_default (v1 in-process)
- IDENTICAL: quirk stop_whitespace_kept (v1 in-process)
- IDENTICAL: quirk thinking_budget_clamped_to_bedrock_min (v1 in-process)
- IDENTICAL: quirk thinking_rewrites_forced_choice_to_auto (v1 in-process)
- IDENTICAL: quirk tool_name_normalized (v1 in-process)
- IDENTICAL: quirk top_k_additional_field (v1 in-process)

## bedrock_converse: responses (snapshot == v1 transform_response == v2)

- IDENTICAL: cache_usage
- IDENTICAL: reasoning
- IDENTICAL: text_basic
- IDENTICAL: tool_use

## bedrock_invoke: responses (snapshot == v1 transform_response == v2)

- IDENTICAL: text_basic
- IDENTICAL: thinking
- IDENTICAL: tool_use

## bedrock_converse: streams (snapshot == real decoder replay == v2 fold, parsed-event seam)

- IDENTICAL: cache_usage_stream
- IDENTICAL: reasoning_stream
- IDENTICAL: text_stream
- IDENTICAL: tool_stream

## bedrock_invoke: streams (snapshot == real decoder replay == v2 fold, parsed-event seam)

- IDENTICAL: text_stream
- IDENTICAL: tool_stream

## gemini: request bodies (characterization snapshot == v1-at-HEAD == v2, canonical JSON)

- IDENTICAL: cache_control_messages
- IDENTICAL: cache_control_tools
- IDENTICAL: full_combo
- IDENTICAL: image_base64
- SKIPPED (corpus): image_url (v1 downloads URL media for google ai studio transforms (network); the vertex_ai gemini route passes the URL through as fileData and is pinned)
- IDENTICAL: max_completion_tokens
- IDENTICAL: multi_turn
- IDENTICAL: params_sampling
- FALLBACK (v1 serves it): pdf_base64 (file/document parts are outside the v2 inbound surface)
- IDENTICAL: plain_text
- IDENTICAL: reasoning_effort_low
- IDENTICAL: response_format_json_object
- IDENTICAL: response_format_json_schema
- IDENTICAL: system_prompt
- IDENTICAL: thinking_enabled
- IDENTICAL: thinking_history_blocks
- IDENTICAL: tools_basic
- IDENTICAL: tools_forced_choice
- IDENTICAL: tools_parallel
- IDENTICAL: tools_streamed_args_roundtrip

## vertex_anthropic: request bodies (characterization snapshot == v1-at-HEAD == v2, canonical JSON)

- IDENTICAL: cache_control_messages
- IDENTICAL: cache_control_tools
- IDENTICAL: full_combo
- IDENTICAL: image_base64
- SKIPPED (corpus): image_url (v1 downloads URL media for claude-on-vertex transforms (network); native anthropic passes the URL through and is pinned)
- IDENTICAL: max_completion_tokens
- IDENTICAL: multi_turn
- IDENTICAL: params_sampling
- FALLBACK (v1 serves it): pdf_base64 (file/document parts are outside the v2 inbound surface)
- IDENTICAL: plain_text
- IDENTICAL: reasoning_effort_low
- IDENTICAL: response_format_json_object
- IDENTICAL: response_format_json_schema
- IDENTICAL: system_prompt
- IDENTICAL: thinking_enabled
- IDENTICAL: thinking_history_blocks
- IDENTICAL: tools_basic
- IDENTICAL: tools_forced_choice
- IDENTICAL: tools_parallel
- IDENTICAL: tools_streamed_args_roundtrip

## vertex_gemini: request bodies (characterization snapshot == v1-at-HEAD == v2, canonical JSON)

- IDENTICAL: cache_control_messages
- IDENTICAL: cache_control_tools
- IDENTICAL: full_combo
- IDENTICAL: image_base64
- IDENTICAL: image_url
- IDENTICAL: max_completion_tokens
- IDENTICAL: multi_turn
- IDENTICAL: params_sampling
- FALLBACK (v1 serves it): pdf_base64 (file/document parts are outside the v2 inbound surface)
- IDENTICAL: plain_text
- IDENTICAL: reasoning_effort_low
- IDENTICAL: response_format_json_object
- IDENTICAL: response_format_json_schema
- IDENTICAL: system_prompt
- IDENTICAL: thinking_enabled
- IDENTICAL: thinking_history_blocks
- IDENTICAL: tools_basic
- IDENTICAL: tools_forced_choice
- IDENTICAL: tools_parallel
- IDENTICAL: tools_streamed_args_roundtrip

## google quirk corpus (v1 in-process reference)

- IDENTICAL: quirk cache_marker_cjk_sublimit (vertex_ai/gemini-2.5-pro)
- IDENTICAL: quirk gemini3_default_temperature_and_level (vertex_ai/gemini-3-pro-preview)
- IDENTICAL: quirk gemini3_studio_forwards_function_call_ids (gemini/gemini-3-pro-preview)
- IDENTICAL: quirk image_url_format_override (vertex_ai/gemini-2.5-pro)
- IDENTICAL: quirk message_name_without_marker (vertex_ai/gemini-2.5-pro)
- IDENTICAL: quirk multi_system_messages_two_parts (vertex_ai/gemini-2.5-pro)
- IDENTICAL: quirk parallel_tool_calls_never_reaches_wire (vertex_ai/gemini-2.5-pro)
- IDENTICAL: quirk reasoning_effort_minimal_model_budget (vertex_ai/gemini-2.5-pro)
- IDENTICAL: quirk stop_as_string (vertex_ai/gemini-2.5-pro)
- IDENTICAL: quirk studio_response_schema_property_ordering (gemini/gemini-exp-1206)
- IDENTICAL: quirk studio_schema_prompt_for_unsupported_model (gemini/gemini-1.5-flash)
- IDENTICAL: quirk studio_top_k_passthrough (gemini/gemini-2.5-flash)
- IDENTICAL: quirk system_only_blank_user_message (vertex_ai/gemini-2.5-pro)
- IDENTICAL: quirk thinking_budget_zero (vertex_ai/gemini-2.5-pro)
- IDENTICAL: quirk tool_choice_none_mode (vertex_ai/gemini-2.5-pro)
- IDENTICAL: quirk tool_without_parameters (vertex_ai/gemini-2.5-pro)
- IDENTICAL: quirk vertex_schema_prompt_for_unsupported_capability (vertex_ai/gemini-pro-latest)
- IDENTICAL: quirk vertex_top_k_passthrough (vertex_ai/gemini-2.5-pro)
- FALLBACK (v1 serves it): cache-marker token bound cjk_under_char_limit (v1's check_and_create_cache may create the context cache; the byte+margin bound fails closed)
- FALLBACK (v1 serves it): cache-marker token bound emoji_under_char_limit (v1's check_and_create_cache may create the context cache; the byte+margin bound fails closed)
- FALLBACK (v1 serves it): cache-marker token bound name_beside_marker (v1's check_and_create_cache may create the context cache; the byte+margin bound fails closed)
- FALLBACK (v1 serves it): cache-marker token bound unmarked_image_beside_marker (v1's check_and_create_cache may create the context cache; the byte+margin bound fails closed)

## gemini: responses (snapshot == v1 transform_response == v2)

- IDENTICAL: text_basic
- IDENTICAL: thinking
- IDENTICAL: tool_use

## vertex_anthropic: responses (snapshot == v1 transform_response == v2)

- IDENTICAL: text_basic
- IDENTICAL: thinking
- IDENTICAL: tool_use

## vertex_gemini: responses (snapshot == v1 transform_response == v2)

- IDENTICAL: text_basic
- IDENTICAL: thinking
- IDENTICAL: tool_use

## gemini: streams (snapshot == real decoder replay == v2 fold)

- IDENTICAL: text_stream
- IDENTICAL: tool_stream

## vertex_anthropic: streams (snapshot == real decoder replay == v2 fold)

- IDENTICAL: text_stream
- IDENTICAL: tool_stream

## vertex_gemini: streams (snapshot == real decoder replay == v2 fold)

- IDENTICAL: text_stream
- IDENTICAL: thinking_stream
- IDENTICAL: tool_stream

Result: 0 divergent rows. Shapes outside the corpus fall back to v1 (fail-closed), so this table is the complete flag-on surface.
