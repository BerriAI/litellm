use litellm_llms_types::formats::messages::{Message, MessagesOptionalParams, MessagesRequest};
use serde_json::{Value, json};

use crate::{
    Error,
    anthropic::common_utils::{
        flatten_unencrypted_web_search_results, sanitize_tool_use_ids, strip_empty_content_blocks,
        strip_provider_specific_fields,
    },
};

pub fn shape_anthropic_messages_request(
    request: MessagesRequest,
) -> Result<MessagesRequest, Error> {
    Ok(MessagesRequest {
        messages: sanitize_anthropic_messages(request.messages),
        params: MessagesOptionalParams {
            metadata: request
                .params
                .metadata
                .as_ref()
                .map(validate_anthropic_api_metadata)
                .transpose()?,
            ..request.params
        },
        ..request
    })
}

fn sanitize_anthropic_messages(messages: Vec<Message>) -> Vec<Message> {
    strip_provider_specific_fields(flatten_unencrypted_web_search_results(
        sanitize_tool_use_ids(strip_empty_content_blocks(messages)),
    ))
}

fn validate_anthropic_api_metadata(metadata: &Value) -> Result<Value, Error> {
    let Value::Object(fields) = metadata else {
        return Err(Error::InvalidRequest(crate::ErrorDetail::InvalidValue {
            field: "metadata",
            expected: "an object",
            actual: metadata.clone(),
        }));
    };
    match fields.get("user_id") {
        None | Some(Value::Null) => Ok(json!({})),
        Some(Value::String(user_id)) => Ok(json!({"user_id": user_id})),
        Some(other) => Err(Error::InvalidRequest(crate::ErrorDetail::InvalidValue {
            field: "metadata.user_id",
            expected: "a string",
            actual: other.clone(),
        })),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn messages(value: Value) -> Vec<Message> {
        serde_json::from_value(value).unwrap()
    }

    fn request(body: Value) -> MessagesRequest {
        serde_json::from_value(body).unwrap()
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
    fn sanitize_anthropic_messages_cleans_replayed_history(
        #[case] history: Value,
        #[case] expected: Value,
    ) {
        assert_eq!(
            serde_json::to_value(sanitize_anthropic_messages(messages(history))).unwrap(),
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
        Err(Error::InvalidRequest("metadata.user_id must be a string, got 123".to_string().into())),
    )]
    #[case::boolean_user_id(
        json!({"user_id": true}),
        Err(Error::InvalidRequest("metadata.user_id must be a string, got true".to_string().into())),
    )]
    #[case::not_an_object(
        json!(["u-1"]),
        Err(Error::InvalidRequest(r#"metadata must be an object, got ["u-1"]"#.to_string().into())),
    )]
    fn validate_anthropic_api_metadata_passes_only_a_string_user_id(
        #[case] metadata: Value,
        #[case] expected: Result<Value, Error>,
    ) {
        assert_eq!(
            validate_anthropic_api_metadata(&metadata).map_err(|error| error.to_string()),
            expected.map_err(|error| error.to_string()),
        );
    }

    #[test]
    fn shaping_cleans_messages_and_metadata() {
        let sanitized = shape_anthropic_messages_request(request(json!({
            "model": "m",
            "messages": [{"role": "assistant", "content": [
                {"type": "text", "text": ""},
                {"type": "tool_use", "id": "functions.Bash:0", "name": "Bash", "input": {}}
            ]}],
            "metadata": {"user_id": "u", "trace_id": "t"},
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "safeguards": [{"type": "dangerous_tool_use"}]
        })))
        .unwrap();
        assert_eq!(
            serde_json::to_value(sanitized).unwrap(),
            json!({
                "model": "m",
                "messages": [{"role": "assistant", "content": [
                    {"type": "tool_use", "id": "functions_Bash_0", "name": "Bash", "input": {}}
                ]}],
                "metadata": {"user_id": "u"},
                "thinking": {"type": "enabled", "budget_tokens": 1024},
                "safeguards": [{"type": "dangerous_tool_use"}]
            })
        );
    }
}
