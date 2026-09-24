use litellm_core_utils::{
    dot_notation_indexing::delete_nested_value,
    get_llm_provider_logic::{CustomLlmProvider, get_custom_llm_provider},
};
use litellm_llms::{
    anthropic::common_utils::{
        flatten_unencrypted_web_search_results, sanitize_tool_use_ids, strip_empty_content_blocks,
        strip_provider_specific_fields,
    },
    base_llm::anthropic_messages::transformation::MessagesTransformContext,
};
use litellm_types::llms::anthropic_messages::anthropic_request::{
    AnthropicMessage, AnthropicMessagesRequest,
};
use serde_json::{Map, Value, json};

use super::{
    Error,
    common_utils::{messages_provider_config, string_headers},
};
use crate::messages::types::{MessagesRequest, MessagesShaping, ProviderMessagesRequest};

pub(super) fn prepare_provider_request(
    request: MessagesRequest<'_>,
) -> Result<ProviderMessagesRequest, Error> {
    let provider_info = get_custom_llm_provider(request.model, request.custom_llm_provider)
        .or_else(|| {
            request
                .custom_llm_provider
                .map(|provider| CustomLlmProvider {
                    model: request.model,
                    custom_llm_provider: provider,
                })
        })
        .ok_or_else(|| {
            Error::InvalidProvider(
                "unable to resolve custom_llm_provider for messages request".to_string(),
            )
        })?;
    let model = provider_info.model.to_string();
    let provider = provider_info.custom_llm_provider;

    let config = messages_provider_config(provider)
        .ok_or_else(|| Error::InvalidProvider(provider.to_string()))?;
    let env_lookup = |key: &str| std::env::var(key).ok();

    let typed_request: AnthropicMessagesRequest =
        serde_json::from_value(request.body).map_err(invalid_request)?;
    let sanitized = sanitize_request(
        AnthropicMessagesRequest {
            model: model.clone(),
            ..typed_request
        },
        &request.shaping,
    )?;
    let trimmed =
        without_additional_drop_params(sanitized, &request.shaping.additional_drop_params)?;
    let transformed = config.transform_anthropic_messages_request(
        trimmed,
        &MessagesTransformContext::new(request.shaping.capabilities, request.shaping.drop_params),
    )?;

    let forwarded = string_headers(request.extra_headers)?;
    let authenticated = config.authenticate(forwarded, request.api_key, &env_lookup)?;
    let headers = config.request_headers(
        with_default_headers(authenticated, config.default_headers()),
        &transformed,
    );

    let body = serde_json::to_value(transformed).map_err(|err| {
        Error::InvalidRequest(format!(
            "failed to serialize Anthropic messages request: {err}"
        ))
    })?;

    let url = config.get_complete_url(request.api_base, &model, &env_lookup)?;

    Ok(ProviderMessagesRequest {
        provider: provider.to_string(),
        model,
        config,
        url,
        body,
        upstream_headers: headers,
        timeout: request.timeout,
    })
}

fn invalid_request(err: serde_json::Error) -> Error {
    Error::InvalidRequest(format!("invalid Anthropic messages request: {err}"))
}

fn without_additional_drop_params(
    request: AnthropicMessagesRequest,
    paths: &[String],
) -> Result<AnthropicMessagesRequest, Error> {
    if paths.is_empty() {
        return Ok(request);
    }
    let Value::Object(fields) = serde_json::to_value(request).map_err(invalid_request)? else {
        return Err(Error::InvalidRequest(
            "Anthropic messages request did not serialize to an object".to_string(),
        ));
    };
    let (required, optional): (Map<String, Value>, Map<String, Value>) = fields
        .into_iter()
        .partition(|(key, _)| matches!(key.as_str(), "model" | "messages"));
    let trimmed = paths.iter().fold(Value::Object(optional), |body, path| {
        delete_nested_value(body, path)
    });
    let merged: Map<String, Value> = required
        .into_iter()
        .chain(trimmed.as_object().cloned().unwrap_or_default())
        .collect();
    serde_json::from_value(Value::Object(merged)).map_err(invalid_request)
}

fn sanitize_request(
    request: AnthropicMessagesRequest,
    shaping: &MessagesShaping,
) -> Result<AnthropicMessagesRequest, Error> {
    Ok(AnthropicMessagesRequest {
        messages: sanitize_messages(request.messages),
        metadata: request
            .metadata
            .as_ref()
            .map(allowed_metadata)
            .transpose()?,
        thinking: with_reasoning_auto_summary(request.thinking, shaping.reasoning_auto_summary),
        ..request
    })
}

fn sanitize_messages(messages: Vec<AnthropicMessage>) -> Vec<AnthropicMessage> {
    strip_provider_specific_fields(flatten_unencrypted_web_search_results(
        sanitize_tool_use_ids(strip_empty_content_blocks(messages)),
    ))
}

fn allowed_metadata(metadata: &Value) -> Result<Value, Error> {
    let Value::Object(fields) = metadata else {
        return Err(Error::InvalidRequest(format!(
            "metadata must be an object, got {metadata}"
        )));
    };
    match fields.get("user_id") {
        None | Some(Value::Null) => Ok(json!({})),
        Some(Value::String(user_id)) => Ok(json!({"user_id": user_id})),
        Some(other) => Err(Error::InvalidRequest(format!(
            "metadata.user_id must be a string, got {other}"
        ))),
    }
}

fn with_reasoning_auto_summary(thinking: Option<Value>, enabled: bool) -> Option<Value> {
    let Some(Value::Object(thinking)) = thinking else {
        return thinking;
    };
    if !enabled || thinking.get("type").and_then(Value::as_str) == Some("disabled") {
        return Some(Value::Object(thinking));
    }
    Some(Value::Object(
        thinking
            .into_iter()
            .filter(|(key, _)| key != "display")
            .chain([("display".to_string(), json!("summarized"))])
            .collect(),
    ))
}

fn with_default_headers(
    headers: Vec<(String, String)>,
    defaults: &[(&str, &str)],
) -> Vec<(String, String)> {
    let missing: Vec<(String, String)> = defaults
        .iter()
        .filter(|(name, _)| {
            !headers
                .iter()
                .any(|(header, _)| header.eq_ignore_ascii_case(name))
        })
        .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
        .collect();
    headers.into_iter().chain(missing).collect()
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};

    use super::*;

    fn messages(value: Value) -> Vec<AnthropicMessage> {
        serde_json::from_value(value).unwrap()
    }

    fn request(body: Value) -> AnthropicMessagesRequest {
        serde_json::from_value(body).unwrap()
    }

    #[fixture]
    fn shaping() -> MessagesShaping {
        MessagesShaping::default()
    }

    fn prepared_body(body: Value, shaping: MessagesShaping) -> Result<Value, Error> {
        prepare_provider_request(MessagesRequest {
            model: "anthropic/claude-test",
            body,
            api_key: Some("sk-test"),
            api_base: Some("https://anthropic.test"),
            custom_llm_provider: Some("anthropic"),
            extra_headers: None,
            timeout: None,
            shaping,
        })
        .map(|prepared| prepared.body)
    }

    #[rstest]
    #[case::empty_text_next_to_a_tool_use(
        json!([{"role": "assistant", "content": [
            {"type": "text", "text": "   "},
            {"type": "tool_use", "id": "t", "name": "B", "input": {}}
        ]}]),
        json!([{"role": "assistant", "content": [
            {"type": "tool_use", "id": "t", "name": "B", "input": {}}
        ]}]),
    )]
    #[case::cross_provider_tool_ids(
        json!([
            {"role": "assistant", "content": [{"type": "tool_use", "id": "functions.Bash:0", "name": "Bash", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "functions.Bash:0", "content": "ok"}]}
        ]),
        json!([
            {"role": "assistant", "content": [{"type": "tool_use", "id": "functions_Bash_0", "name": "Bash", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "functions_Bash_0", "content": "ok"}]}
        ]),
    )]
    #[case::replayed_unencrypted_web_search_results(
        json!([
            {"role": "user", "content": "latest litellm version?"},
            {"role": "assistant", "content": [
                {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "latest litellm version"}},
                {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [{
                    "type": "web_search_result",
                    "url": "https://github.com/BerriAI/litellm/releases",
                    "title": "Releases",
                    "page_age": null,
                    "encrypted_content": "",
                    "snippet": "Latest release v1.95.0"
                }]}
            ]},
            {"role": "user", "content": "which version?"}
        ]),
        json!([
            {"role": "user", "content": "latest litellm version?"},
            {"role": "assistant", "content": [{
                "type": "text",
                "text": "Web search results for 'latest litellm version':\n\nTitle: Releases\nURL: https://github.com/BerriAI/litellm/releases\nSnippet: Latest release v1.95.0"
            }]},
            {"role": "user", "content": "which version?"}
        ]),
    )]
    #[case::replayed_provider_specific_fields(
        json!([
            {"role": "assistant", "content": [{
                "type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "Paris"},
                "provider_specific_fields": {"signature": "sig_abc"}
            }]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "Sunny"}]}
        ]),
        json!([
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "Paris"}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "Sunny"}]}
        ]),
    )]
    #[case::ids_are_normalized_before_web_search_results_flatten(
        json!([
            {"role": "user", "content": "run it"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "", "signature": "sig"},
                {"type": "text", "text": ""},
                {"type": "tool_use", "id": "functions.Bash:0", "name": "Bash", "input": {}, "provider_specific_fields": {"x": 1}},
                {"type": "server_tool_use", "id": "srv.1", "name": "web_search", "input": {"query": "q"}, "provider_specific_fields": {"x": 2}},
                {"type": "web_search_tool_result", "tool_use_id": "srv.1", "provider_specific_fields": {"x": 3}, "content": [
                    {"type": "web_search_result", "url": "u", "title": "", "encrypted_content": "", "provider_specific_fields": {"x": 4}}
                ]}
            ]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "functions.Bash:0", "content": "ok"}]},
            {"role": "assistant", "content": [{"type": "text", "text": " "}]}
        ]),
        json!([
            {"role": "user", "content": "run it"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "functions_Bash_0", "name": "Bash", "input": {}},
                {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {"query": "q"}},
                {"type": "text", "text": "Web search results:\n\nURL: u"}
            ]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "functions_Bash_0", "content": "ok"}]}
        ]),
    )]
    fn sanitize_messages_cleans_replayed_history(#[case] history: Value, #[case] expected: Value) {
        assert_eq!(
            serde_json::to_value(sanitize_messages(messages(history))).unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::keeps_only_user_id(json!({"user_id": "u-1", "trace_id": "internal"}), Ok(json!({"user_id": "u-1"})))]
    #[case::null_user_id(json!({"user_id": null, "trace_id": "internal"}), Ok(json!({})))]
    #[case::no_user_id(json!({"trace_id": "internal"}), Ok(json!({})))]
    #[case::empty(json!({}), Ok(json!({})))]
    #[case::numeric_user_id(
        json!({"user_id": 123}),
        Err(Error::InvalidRequest("metadata.user_id must be a string, got 123".to_string())),
    )]
    #[case::boolean_user_id(
        json!({"user_id": true}),
        Err(Error::InvalidRequest("metadata.user_id must be a string, got true".to_string())),
    )]
    #[case::not_an_object(
        json!(["u-1"]),
        Err(Error::InvalidRequest(r#"metadata must be an object, got ["u-1"]"#.to_string())),
    )]
    fn allowed_metadata_passes_only_a_string_user_id(
        #[case] metadata: Value,
        #[case] expected: Result<Value, Error>,
    ) {
        assert_eq!(allowed_metadata(&metadata), expected);
    }

    #[rstest]
    #[case::adaptive(
        Some(json!({"type": "adaptive", "budget_tokens": 5000})),
        true,
        Some(json!({"type": "adaptive", "budget_tokens": 5000, "display": "summarized"})),
    )]
    #[case::enabled(
        Some(json!({"type": "enabled", "budget_tokens": 10000})),
        true,
        Some(json!({"type": "enabled", "budget_tokens": 10000, "display": "summarized"})),
    )]
    #[case::no_type(Some(json!({})), true, Some(json!({"display": "summarized"})))]
    #[case::display_omitted_is_overridden(
        Some(json!({"type": "enabled", "budget_tokens": 10000, "display": "omitted"})),
        true,
        Some(json!({"type": "enabled", "budget_tokens": 10000, "display": "summarized"})),
    )]
    #[case::display_summarized_is_kept(
        Some(json!({"type": "enabled", "display": "summarized"})),
        true,
        Some(json!({"type": "enabled", "display": "summarized"})),
    )]
    #[case::disabled_thinking(Some(json!({"type": "disabled"})), true, Some(json!({"type": "disabled"})))]
    #[case::flag_off(
        Some(json!({"type": "enabled", "budget_tokens": 10000})),
        false,
        Some(json!({"type": "enabled", "budget_tokens": 10000})),
    )]
    #[case::flag_off_keeps_callers_display(
        Some(json!({"type": "enabled", "display": "omitted"})),
        false,
        Some(json!({"type": "enabled", "display": "omitted"})),
    )]
    #[case::no_thinking(None, true, None)]
    #[case::non_object_thinking(Some(json!("enabled")), true, Some(json!("enabled")))]
    fn reasoning_auto_summary_marks_active_thinking_as_summarized(
        #[case] thinking: Option<Value>,
        #[case] enabled: bool,
        #[case] expected: Option<Value>,
    ) {
        assert_eq!(with_reasoning_auto_summary(thinking, enabled), expected);
    }

    #[rstest]
    #[case::nothing_forwarded(
        &[],
        &[("x-version", "1"), ("content-type", "application/json")],
        &[("x-version", "1"), ("content-type", "application/json")],
    )]
    #[case::forwarded_header_wins_in_any_case(
        &[("X-Version", "custom"), ("x-api-key", "k")],
        &[("x-version", "1"), ("content-type", "application/json")],
        &[("X-Version", "custom"), ("x-api-key", "k"), ("content-type", "application/json")],
    )]
    #[case::no_defaults(&[("x-api-key", "k")], &[], &[("x-api-key", "k")])]
    fn default_headers_fill_only_missing_names(
        #[case] forwarded: &[(&str, &str)],
        #[case] defaults: &[(&str, &str)],
        #[case] expected: &[(&str, &str)],
    ) {
        let owned = |headers: &[(&str, &str)]| -> Vec<(String, String)> {
            headers
                .iter()
                .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
                .collect()
        };
        assert_eq!(
            with_default_headers(owned(forwarded), defaults),
            owned(expected)
        );
    }

    #[rstest]
    fn sanitize_request_shapes_messages_metadata_and_thinking(shaping: MessagesShaping) {
        let sanitized = sanitize_request(
            request(json!({
                "model": "m",
                "messages": [{"role": "assistant", "content": [
                    {"type": "text", "text": ""},
                    {"type": "tool_use", "id": "functions.Bash:0", "name": "Bash", "input": {}}
                ]}],
                "metadata": {"user_id": "u", "trace_id": "t"},
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "safeguards": [{"type": "dangerous_tool_use"}]
            })),
            &MessagesShaping {
                reasoning_auto_summary: true,
                ..shaping
            },
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(sanitized).unwrap(),
            json!({
                "model": "m",
                "messages": [{"role": "assistant", "content": [
                    {"type": "tool_use", "id": "functions_Bash_0", "name": "Bash", "input": {}}
                ]}],
                "metadata": {"user_id": "u"},
                "thinking": {"type": "enabled", "budget_tokens": 1024, "display": "summarized"},
                "safeguards": [{"type": "dangerous_tool_use"}]
            })
        );
    }

    #[rstest]
    #[case::top_level_and_nested_paths(
        json!({
            "max_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "context_management": {"edits": [{"type": "clear_thinking_20251015"}]},
            "metadata": {"user_id": "u1"},
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}, "input_examples": [{"q": "x"}]}]
        }),
        &["thinking", "context_management", "tools[*].input_examples"],
        json!({
            "max_tokens": 1024,
            "metadata": {"user_id": "u1"},
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}}]
        }),
    )]
    #[case::no_paths(
        json!({"max_tokens": 16, "safeguards": [{"type": "dangerous_tool_use"}]}),
        &[],
        json!({"max_tokens": 16, "safeguards": [{"type": "dangerous_tool_use"}]}),
    )]
    #[case::model_and_messages_are_never_dropped(
        json!({"max_tokens": 16}),
        &["model", "messages", "messages[0].content"],
        json!({"max_tokens": 16}),
    )]
    fn prepared_body_drops_configured_paths(
        shaping: MessagesShaping,
        #[case] fields: Value,
        #[case] additional_drop_params: &[&str],
        #[case] expected_fields: Value,
    ) {
        let with_messages = |fields: Value| -> Value {
            let Value::Object(fields) = fields else {
                unreachable!()
            };
            Value::Object(
                [
                    ("model".to_string(), json!("claude-test")),
                    (
                        "messages".to_string(),
                        json!([{"role": "user", "content": "hi"}]),
                    ),
                ]
                .into_iter()
                .chain(fields)
                .collect(),
            )
        };
        let shaping = MessagesShaping {
            additional_drop_params: additional_drop_params
                .iter()
                .map(ToString::to_string)
                .collect(),
            ..shaping
        };
        assert_eq!(
            prepared_body(with_messages(fields), shaping),
            Ok(with_messages(expected_fields))
        );
    }

    #[rstest]
    fn prepared_body_carries_the_provider_stripped_model(shaping: MessagesShaping) {
        assert_eq!(
            prepared_body(
                json!({
                    "model": "anthropic/claude-test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16
                }),
                shaping,
            ),
            Ok(json!({
                "model": "claude-test",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16
            }))
        );
    }

    #[rstest]
    fn dropped_thinking_display_is_not_restored_by_auto_summary(shaping: MessagesShaping) {
        let shaping = MessagesShaping {
            reasoning_auto_summary: true,
            additional_drop_params: vec!["thinking.display".to_string()],
            ..shaping
        };
        assert_eq!(
            prepared_body(
                json!({
                    "model": "claude-test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 4096,
                    "thinking": {"type": "enabled", "budget_tokens": 2048}
                }),
                shaping,
            ),
            Ok(json!({
                "model": "claude-test",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4096,
                "thinking": {"type": "enabled", "budget_tokens": 2048}
            }))
        );
    }

    #[rstest]
    fn dropping_an_invalid_metadata_user_id_does_not_skip_its_validation(shaping: MessagesShaping) {
        let shaping = MessagesShaping {
            additional_drop_params: vec!["metadata.user_id".to_string()],
            ..shaping
        };
        assert!(matches!(
            prepared_body(
                json!({
                    "model": "claude-test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16,
                    "metadata": {"user_id": 123}
                }),
                shaping,
            ),
            Err(Error::InvalidRequest(_))
        ));
    }

    #[rstest]
    fn prepared_body_rejects_invalid_metadata_before_the_call(shaping: MessagesShaping) {
        assert_eq!(
            prepared_body(
                json!({
                    "model": "claude-test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 16,
                    "metadata": {"user_id": 123}
                }),
                shaping,
            ),
            Err(Error::InvalidRequest(
                "metadata.user_id must be a string, got 123".to_string()
            ))
        );
    }
}
