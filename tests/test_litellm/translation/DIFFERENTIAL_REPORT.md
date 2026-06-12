# Translation v2 differential report (anthropic + bedrock + openai)

v1 and v2 run over the same corpus; every row must be IDENTICAL (or an
explained FALLBACK that v1 serves) for a provider's flag to turn on.
Bedrock rows additionally pin the characterization-corpus snapshot, so
each row proves snapshot == v1-at-HEAD == v2. Regenerate with:
`python -m tests.test_litellm.translation.generate_differential_report`

- commit: c1fee272a0

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
- FALLBACK (v1 serves it): tool_call_compact_arguments (non-canonical JSON spacing)
- FALLBACK (v1 serves it): top_k_not_openai (top_k)
- FALLBACK (v1 serves it): user_param_model_list_gate (open_ai_chat_completion_models)

## openai_compat: responses (v1 convert_to_model_response_object vs v2)

- IDENTICAL: cached_and_reasoning_usage_details
- IDENTICAL: reasoning_content_key
- IDENTICAL: text
- IDENTICAL: think_tag_extraction
- IDENTICAL: tool_calls_rewrites_stop

## openai_compat: streams (v1 CustomStreamWrapper over SDK chunks vs v2 fold)

- IDENTICAL: empty_keepalive_swallowed
- IDENTICAL: text
- IDENTICAL: text_no_leading_role
- IDENTICAL: tools
- SEAM CONTRACT: usage tail (v2 passes the wire choices=[] usage chunk through; v1's wrapper synthesizes its final usage chunk from it, which is the streaming seam's envelope to reproduce)

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

Result: 0 divergent rows. Shapes outside the corpus fall back to v1 (fail-closed), so this table is the complete flag-on surface.
