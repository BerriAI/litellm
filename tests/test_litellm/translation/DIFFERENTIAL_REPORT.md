# Translation v2 differential report (anthropic)

v1 and v2 run over the same corpus; every row must be IDENTICAL for
the anthropic flag to turn on. Regenerate with:
`python -m tests.test_litellm.translation.generate_differential_report`

- commit: 23724c4392

## Request bodies (v1 map_openai_params + transform_request vs v2)

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

## Responses (v1 transform_response vs v2)

- IDENTICAL: json_tool
- IDENTICAL: text
- IDENTICAL: thinking
- IDENTICAL: tools

## Streams (v1 CustomStreamWrapper replay vs v2 engine/stream)

- IDENTICAL: text
- IDENTICAL: thinking
- IDENTICAL: tools

Result: 0 divergent rows. Shapes outside the corpus fall back to v1 (fail-closed), so this table is the complete flag-on surface.
