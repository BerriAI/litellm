//! What the standard logging payload logs of a Messages call's request, kept on the call's record
//! once the request is read so the body itself need not outlive its send.

use litellm_llms_types::formats::messages::MessagesOptionalParams;
use litellm_standard_logging::{messages_model_parameters, truncate_base64, with_system_prompt};
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

/// The params `get_requested_anthropic_messages_optional_param` drops for a model or provider
/// that does not take them, under `drop_params`.
const GATED_PARAMS: [&str; 5] = ["speed", "service_tier", "temperature", "top_p", "top_k"];

/// The request as the standard logging payload logs it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LoggedRequest {
    /// The payload's `messages`, long base64 payloads cut: the attempt's messages with its
    /// `system` prompt in front, or the client's messages as sent for a call refused before its
    /// first attempt.
    pub messages: Value,
    /// The payload's `model_parameters`: the attempt's optional params less the prompt ones, or
    /// the client's `stream` alone for a call refused before its first attempt.
    pub model_parameters: Map<String, Value>,
    /// The metadata's `requester_metadata`: the body's `metadata` object as the client sent it.
    pub requester_metadata: Option<Map<String, Value>>,
}

impl LoggedRequest {
    /// Reads what Python's proxy takes of the request before any hook changes it.
    /// `requester_metadata` is the snapshot `add_litellm_data_to_request` keeps of a `metadata`
    /// object.
    ///
    /// The messages and model params are what `ProxyLogging._handle_logging_proxy_only_error`
    /// logs for a call refused before it reaches the provider: of the body's fields it hands the
    /// logging object only `_PROXY_ONLY_ERROR_LOGGED_FIELDS`, so the messages are the body's
    /// without its `system` prompt, which never reaches the logged kwargs, and the optional params
    /// are the body's `stream` when it sends one. An attempt overwrites both. A base64 payload
    /// longer than `max_base64_length` is cut, none when it is 0.
    pub fn sent(&mut self, body: &Map<String, Value>, max_base64_length: usize) {
        self.messages = truncate_base64(
            body.get("messages").cloned().unwrap_or(Value::Null),
            max_base64_length,
        );
        self.model_parameters = messages_model_parameters(
            body.get("stream")
                .map(|stream| ("stream".to_owned(), stream.clone()))
                .into_iter()
                .collect(),
        );
        self.requester_metadata = body.get("metadata").and_then(Value::as_object).cloned();
    }

    /// Reads the messages and optional params of the body an attempt sends, `drops` naming the
    /// call's `additional_drop_params`. Each attempt overwrites the last, as each of Python's
    /// `update_from_kwargs` does.
    pub fn attempted(
        &mut self,
        body: &Map<String, Value>,
        drops: &[String],
        reasoning_auto_summary: bool,
        max_base64_length: usize,
    ) {
        let params = optional_params(body, drops, reasoning_auto_summary);
        let messages = body.get("messages").cloned().unwrap_or(Value::Null);
        self.messages = truncate_base64(
            with_system_prompt(messages, params.get("system")),
            max_base64_length,
        );
        self.model_parameters = messages_model_parameters(params);
    }

    /// Leaves out of the model parameters each param the attempt's route dropped for its model
    /// or provider, `sent` being the optional params the route sends
    /// (`RequestContext.optional_params`), since Python logs the params after that gate.
    pub fn gated(&mut self, sent: &Value) {
        self.model_parameters
            .retain(|name, _| !GATED_PARAMS.contains(&name.as_str()) || sent.get(name).is_some());
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
    use litellm_standard_logging::MAX_BASE64_LENGTH_FOR_LOGGING;
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
        logged.attempted(
            &object(body),
            &drops,
            reasoning_auto_summary,
            MAX_BASE64_LENGTH_FOR_LOGGING,
        );
        logged
    }

    #[rstest]
    #[case::metadata(json!({"metadata": {"user_id": " user-1 "}}), Some(json!({"user_id": " user-1 "})))]
    #[case::empty_metadata(json!({"metadata": {}}), Some(json!({})))]
    #[case::null_metadata(json!({"metadata": null}), None)]
    #[case::no_metadata(json!({}), None)]
    fn the_requester_metadata_is_the_bodys_metadata(
        #[case] body: Value,
        #[case] requester_metadata: Option<Value>,
    ) {
        let mut logged = LoggedRequest::default();
        logged.sent(&object(body), MAX_BASE64_LENGTH_FOR_LOGGING);
        assert_eq!(logged.requester_metadata, requester_metadata.map(object));
    }

    #[rstest]
    #[case::stream_sent(
        json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "system": "s", "max_tokens": 8, "stream": true}),
        json!([{"role": "user", "content": "hi"}]),
        json!({"stream": true})
    )]
    #[case::stream_left_out(
        json!({"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, "top_k": 3}),
        json!([{"role": "user", "content": "hi"}]),
        json!({})
    )]
    #[case::stream_null(json!({"messages": "hi", "stream": null}), json!("hi"), json!({"stream": null}))]
    #[case::no_messages(json!({"model": "m"}), Value::Null, json!({}))]
    fn a_call_refused_before_an_attempt_logs_the_bodys_messages_and_stream(
        #[case] body: Value,
        #[case] messages: Value,
        #[case] model_parameters: Value,
    ) {
        let mut logged = LoggedRequest::default();
        logged.sent(&object(body), MAX_BASE64_LENGTH_FOR_LOGGING);
        assert_eq!(
            (logged.messages, Value::Object(logged.model_parameters)),
            (messages, model_parameters)
        );
    }

    #[rstest]
    fn long_base64_in_the_sent_messages_is_cut() {
        let image = json!({"type": "image", "source": {"type": "base64", "data": "A".repeat(100)}});
        let mut logged = LoggedRequest::default();
        logged.sent(
            &object(json!({"messages": [{"role": "user", "content": [image]}]})),
            MAX_BASE64_LENGTH_FOR_LOGGING,
        );
        assert_eq!(
            logged.messages,
            json!([{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "data": "[base64_data truncated: 75B]"}}
            ]}])
        );
    }

    #[rstest]
    fn an_attempt_overwrites_what_the_sent_body_logged() {
        let body = object(json!({
            "messages": [{"role": "user", "content": "hi"}],
            "system": "s",
            "max_tokens": 8,
        }));
        let mut logged = LoggedRequest::default();
        logged.sent(&body, MAX_BASE64_LENGTH_FOR_LOGGING);
        logged.attempted(&body, &[], false, MAX_BASE64_LENGTH_FOR_LOGGING);
        assert_eq!(
            (logged.messages, Value::Object(logged.model_parameters)),
            (
                json!([{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]),
                json!({"max_tokens": 8, "stream": false})
            )
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
    #[case::sampling_dropped_for_the_model(
        json!({"max_tokens": 8, "stream": true, "temperature": 0.5, "top_p": 0.9, "top_k": 3}),
        json!({"max_tokens": 8, "stream": true}),
        json!({"max_tokens": 8, "stream": true})
    )]
    #[case::speed_and_service_tier_dropped(
        json!({"max_tokens": 8, "stream": true, "speed": "fast", "service_tier": "auto"}),
        json!({"max_tokens": 8, "stream": true}),
        json!({"max_tokens": 8, "stream": true})
    )]
    #[case::temperature_one_kept(
        json!({"max_tokens": 8, "stream": true, "temperature": 1.0, "top_p": 0.9}),
        json!({"max_tokens": 8, "stream": true, "temperature": 1.0}),
        json!({"max_tokens": 8, "stream": true, "temperature": 1.0})
    )]
    #[case::the_default_stream_kept(
        json!({"max_tokens": 8, "temperature": 0.5}),
        json!({"max_tokens": 8, "temperature": 0.5}),
        json!({"max_tokens": 8, "stream": false, "temperature": 0.5})
    )]
    fn the_model_parameters_leave_out_what_the_route_dropped_for_the_model(
        #[case] body: Value,
        #[case] sent: Value,
        #[case] expected: Value,
    ) {
        let mut logged = attempted(body, &[], false);
        logged.gated(&sent);
        assert_eq!(Value::Object(logged.model_parameters), expected);
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

    #[rstest]
    #[case::a_longer_limit(100, "A".repeat(100))]
    #[case::no_limit(0, "A".repeat(100))]
    #[case::a_shorter_limit(10, "[base64_data truncated: 75B]".to_owned())]
    fn the_base64_logging_limit_is_the_configured_one(
        #[case] max_base64_length: usize,
        #[case] logged_data: String,
    ) {
        let image =
            |data: &str| json!({"type": "image", "source": {"type": "base64", "data": data}});
        let body =
            object(json!({"messages": [{"role": "user", "content": [image(&"A".repeat(100))]}]}));
        let mut logged = LoggedRequest::default();
        logged.sent(&body, max_base64_length);
        let sent = logged.messages.clone();
        logged.attempted(&body, &[], false, max_base64_length);
        let expected = json!([{"role": "user", "content": [image(&logged_data)]}]);
        assert_eq!((sent, logged.messages), (expected.clone(), expected));
    }
}
