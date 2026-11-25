//! What the standard logging payload logs of a Messages call's request, kept on the call's record
//! once the request is read so the body itself need not outlive its send.

use litellm_llms_types::formats::messages::MessagesOptionalParams;
use litellm_python_compat::unicode::isspace;
use litellm_standard_logging::{
    MAX_BASE64_LENGTH_FOR_LOGGING, messages_model_parameters, truncate_base64, with_system_prompt,
};
use serde_json::{Map, Value};

/// The named params of Python's `anthropic_messages_handler`, in its signature's order, which
/// `locals()` lists them in ahead of the call's other kwargs.
const NAMED_PARAMS: [&str; 12] = [
    "max_tokens",
    "metadata",
    "stop_sequences",
    "stream",
    "system",
    "temperature",
    "thinking",
    "tool_choice",
    "tools",
    "top_k",
    "top_p",
    "container",
];

/// The sampling params `get_requested_anthropic_messages_optional_param` pops and sets again for
/// the model, in that order, which moves them after the others.
const SAMPLING_PARAMS: [&str; 3] = ["temperature", "top_p", "top_k"];

/// The request as the standard logging payload logs it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LoggedRequest {
    /// The payload's `messages`: the attempt's messages with its `system` prompt in front and
    /// long base64 payloads cut, `null` before any attempt.
    pub messages: Value,
    /// The payload's `model_parameters`: the attempt's optional params less the prompt ones.
    pub model_parameters: Map<String, Value>,
    /// The metadata's `requester_metadata`: the body's `metadata` object as the client sent it.
    pub requester_metadata: Option<Map<String, Value>>,
    /// The payload's `end_user`: the body's `metadata.user_id`, the one end user source the
    /// gateway serves a call with.
    pub end_user: Option<String>,
}

impl LoggedRequest {
    /// Reads what Python's proxy takes of the request before any hook changes it.
    /// `requester_metadata` is the snapshot `add_litellm_data_to_request` keeps of a `metadata`
    /// object, and the end user is `get_end_user_id_from_request_body`'s `metadata.user_id`,
    /// stripped and kept when text is left (`_coerce_user_id_to_str`).
    pub fn sent(&mut self, body: &Map<String, Value>) {
        let metadata = body.get("metadata").and_then(Value::as_object);
        self.requester_metadata = metadata.cloned();
        self.end_user = metadata
            .and_then(|metadata| metadata.get("user_id"))
            .and_then(Value::as_str)
            .map(|user_id| user_id.trim_matches(isspace))
            .filter(|user_id| !user_id.is_empty())
            .map(str::to_owned);
    }

    /// Reads the messages and optional params of the body an attempt sends, `drops` naming the
    /// call's `additional_drop_params`. Each attempt overwrites the last, as each of Python's
    /// `update_from_kwargs` does.
    pub fn attempted(
        &mut self,
        body: &Map<String, Value>,
        drops: &[String],
        reasoning_auto_summary: bool,
    ) {
        let params = optional_params(body, drops, reasoning_auto_summary);
        let messages = body.get("messages").cloned().unwrap_or(Value::Null);
        self.messages = truncate_base64(
            with_system_prompt(messages, params.get("system")),
            MAX_BASE64_LENGTH_FOR_LOGGING,
        );
        self.model_parameters = messages_model_parameters(params);
    }
}

/// The `optional_params` Python logs for a Messages call, from
/// `AnthropicMessagesRequestUtils.get_requested_anthropic_messages_optional_param` over the
/// handler's `locals()`: each Messages optional param the body sets to a value other than `null`
/// and `drops` does not name, with `stream` `false` when the body leaves it out (the handler's
/// default), the sampling params last, and `display: "summarized"` in a `thinking` object that is
/// not `disabled` when `reasoning_auto_summary` is on.
fn optional_params(
    body: &Map<String, Value>,
    drops: &[String],
    reasoning_auto_summary: bool,
) -> Map<String, Value> {
    let named = NAMED_PARAMS.iter().filter_map(|&name| {
        let value = match body.get(name) {
            Some(value) => value.clone(),
            None if name == "stream" => Value::Bool(false),
            None => return None,
        };
        Some((name.to_owned(), value))
    });
    let others = body
        .iter()
        .filter(|(name, _)| !NAMED_PARAMS.contains(&name.as_str()))
        .map(|(name, value)| (name.clone(), value.clone()));
    let (sampling, rest): (Vec<_>, Vec<_>) = named
        .chain(others)
        .filter(|(name, value)| {
            !value.is_null()
                && !drops.iter().any(|dropped| dropped == name)
                && MessagesOptionalParams::fields().any(|field| field == name)
        })
        .partition(|(name, _)| SAMPLING_PARAMS.contains(&name.as_str()));
    let sampling = SAMPLING_PARAMS
        .iter()
        .filter_map(|&param| sampling.iter().find(|(name, _)| name == param).cloned());
    let mut params: Map<String, Value> = rest.into_iter().chain(sampling).collect();
    if reasoning_auto_summary
        && let Some(Value::Object(thinking)) = params.get_mut("thinking")
        && thinking.get("type").and_then(Value::as_str) != Some("disabled")
    {
        thinking.insert("display".to_owned(), Value::from("summarized"));
    }
    params
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn object(value: Value) -> Map<String, Value> {
        let Value::Object(fields) = value else {
            panic!("an object");
        };
        fields
    }

    fn attempted(body: Value, drops: &[&str], reasoning_auto_summary: bool) -> LoggedRequest {
        let drops: Vec<String> = drops.iter().map(|&drop| drop.to_owned()).collect();
        let mut logged = LoggedRequest::default();
        logged.attempted(&object(body), &drops, reasoning_auto_summary);
        logged
    }

    #[rstest]
    #[case::metadata(
        json!({"metadata": {"user_id": " user-1 "}}),
        Some(json!({"user_id": " user-1 "})),
        Some("user-1")
    )]
    #[case::blank_user_id(json!({"metadata": {"user_id": "  "}}), Some(json!({"user_id": "  "})), None)]
    #[case::empty_metadata(json!({"metadata": {}}), Some(json!({})), None)]
    #[case::null_metadata(json!({"metadata": null}), None, None)]
    #[case::no_metadata(json!({}), None, None)]
    fn the_requester_metadata_and_end_user_are_the_bodys_metadata(
        #[case] body: Value,
        #[case] requester_metadata: Option<Value>,
        #[case] end_user: Option<&str>,
    ) {
        let mut logged = LoggedRequest::default();
        logged.sent(&object(body));
        assert_eq!(
            (logged.requester_metadata, logged.end_user.as_deref()),
            (requester_metadata.map(object), end_user)
        );
    }

    #[rstest]
    #[case::stream_left_out(
        json!({"model": "m", "messages": [], "max_tokens": 8}),
        &[],
        json!({"max_tokens": 8, "stream": false})
    )]
    #[case::stream_null(
        json!({"model": "m", "messages": [], "max_tokens": 8, "stream": null}),
        &[],
        json!({"max_tokens": 8})
    )]
    #[case::dropped(
        json!({"messages": [], "max_tokens": 8, "stream": true, "top_k": 3}),
        &["top_k", "metadata.user_id"],
        json!({"max_tokens": 8, "stream": true})
    )]
    #[case::prompt_and_metadata_left_out(
        json!({"messages": [], "system": "s", "metadata": {"user_id": "u"}, "stream": true}),
        &[],
        json!({"stream": true})
    )]
    #[case::other_params_after_the_named(
        json!({"output_config": {"effort": "low"}, "messages": [], "max_tokens": 8, "stream": true}),
        &[],
        json!({"max_tokens": 8, "stream": true, "output_config": {"effort": "low"}})
    )]
    fn the_model_parameters_are_the_attempts_optional_params(
        #[case] body: Value,
        #[case] drops: &[&str],
        #[case] expected: Value,
    ) {
        assert_eq!(
            Value::Object(attempted(body, drops, false).model_parameters),
            expected
        );
    }

    #[rstest]
    fn the_sampling_params_come_last_in_their_own_order() {
        let logged = attempted(
            json!({"top_k": 3, "temperature": 0.5, "max_tokens": 8, "top_p": 0.9, "stream": true}),
            &[],
            false,
        );
        assert_eq!(
            logged.model_parameters.keys().collect::<Vec<_>>(),
            ["max_tokens", "stream", "temperature", "top_p", "top_k"]
        );
    }

    #[rstest]
    #[case::enabled(json!({"type": "enabled", "budget_tokens": 8}), true, json!({"type": "enabled", "budget_tokens": 8, "display": "summarized"}))]
    #[case::disabled(json!({"type": "disabled"}), true, json!({"type": "disabled"}))]
    #[case::setting_off(json!({"type": "adaptive"}), false, json!({"type": "adaptive"}))]
    fn reasoning_auto_summary_asks_for_summarized_thinking(
        #[case] thinking: Value,
        #[case] reasoning_auto_summary: bool,
        #[case] expected: Value,
    ) {
        let logged = attempted(
            json!({"thinking": thinking, "stream": true}),
            &[],
            reasoning_auto_summary,
        );
        assert_eq!(logged.model_parameters.get("thinking"), Some(&expected));
    }

    #[rstest]
    #[case::kept(&[], json!([{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]))]
    #[case::dropped(&["system"], json!([{"role": "user", "content": "hi"}]))]
    fn the_messages_carry_the_system_prompt_the_attempt_sends(
        #[case] drops: &[&str],
        #[case] expected: Value,
    ) {
        let logged = attempted(
            json!({"messages": [{"role": "user", "content": "hi"}], "system": "s"}),
            drops,
            false,
        );
        assert_eq!(logged.messages, expected);
    }

    #[rstest]
    fn long_base64_in_the_messages_is_cut() {
        let image = json!({"type": "image", "source": {"type": "base64", "data": "A".repeat(100)}});
        let logged = attempted(
            json!({"messages": [{"role": "user", "content": [image]}]}),
            &[],
            false,
        );
        assert_eq!(
            logged.messages,
            json!([{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "data": "[base64_data truncated: 75B]"}}
            ]}])
        );
    }
}
