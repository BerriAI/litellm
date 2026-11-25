//! The `hidden_params.additional_headers` of a payload: the response headers Python keeps on the
//! answer it logs, or on the exception a failed call raised, as
//! `StandardLoggingPayloadSetup.get_additional_headers` cleans them.

use serde_json::{Map, Value};

/// The keys of `StandardLoggingAdditionalHeaders`, in the order the TypedDict declares them, each
/// read from the header whose name is the key lowercased with `-` for `_`. The first four are
/// typed `int`, the rest `str`, but `get_additional_headers` tries `int()` on every one.
const TYPED_KEYS: [&str; 6] = [
    "x_ratelimit_limit_requests",
    "x_ratelimit_limit_tokens",
    "x_ratelimit_remaining_requests",
    "x_ratelimit_remaining_tokens",
    "x_ratelimit_reset_requests",
    "x_ratelimit_reset_tokens",
];

/// The Anthropic rate limit headers `process_anthropic_headers` copies under their OpenAI names,
/// in the order it copies them.
const ANTHROPIC_RATE_LIMITS: [(&str, &str); 4] = [
    (
        "anthropic-ratelimit-requests-limit",
        "x-ratelimit-limit-requests",
    ),
    (
        "anthropic-ratelimit-requests-remaining",
        "x-ratelimit-remaining-requests",
    ),
    (
        "anthropic-ratelimit-tokens-limit",
        "x-ratelimit-limit-tokens",
    ),
    (
        "anthropic-ratelimit-tokens-remaining",
        "x-ratelimit-remaining-tokens",
    ),
];

/// The prefix `process_anthropic_headers` names every provider header under.
const PROVIDER_PREFIX: &str = "llm_provider-";

/// Sets `name` to `value` in `headers`, in place when it is already there, as a Python `dict`
/// keeps a key where it was first written.
fn put(headers: &mut Vec<(String, String)>, name: String, value: String) {
    match headers.iter_mut().find(|(held, _)| *held == name) {
        Some((_, held)) => *held = value,
        None => headers.push((name, value)),
    }
}

/// `dict(httpx.Headers(headers))`: lowercase names, the values of a repeated name joined with
/// `, ` where the name first came.
pub fn httpx_items(headers: &[(String, String)]) -> Vec<(String, String)> {
    let mut joined: Vec<(String, String)> = Vec::new();
    for (name, value) in headers {
        let name = name.to_ascii_lowercase();
        match joined.iter_mut().find(|(held, _)| *held == name) {
            Some((_, held)) => {
                held.push_str(", ");
                held.push_str(value);
            }
            None => joined.push((name, value.clone())),
        }
    }
    joined
}

/// `process_anthropic_headers(dict(raw_response.headers))`, the headers
/// `AnthropicConfig.transform_response` keeps on the answer it builds: every provider header
/// under `llm_provider-`, then the Anthropic rate limits under their OpenAI names.
pub fn anthropic_response_headers(provider_headers: &[(String, String)]) -> Vec<(String, String)> {
    let raw = httpx_items(provider_headers);
    let mut headers: Vec<(String, String)> = raw
        .iter()
        .map(|(name, value)| (format!("{PROVIDER_PREFIX}{name}"), value.clone()))
        .collect();
    for (anthropic, openai) in ANTHROPIC_RATE_LIMITS {
        if let Some((_, value)) = raw.iter().find(|(name, _)| name == anthropic) {
            put(&mut headers, openai.to_owned(), value.clone());
        }
    }
    headers
}

/// `int(value)` where Python's succeeds, else the text as it is.
fn int_or_text(value: &str) -> Value {
    litellm_python_compat::number::int(value)
        .and_then(|parsed| i64::try_from(parsed).ok())
        .map_or_else(|| Value::from(value), Value::from)
}

/// `StandardLoggingPayloadSetup.get_additional_headers(headers)`: the typed keys first, each
/// read from its header and made an `int` where `int()` takes it, then every other header as it
/// is, in its own order.
pub fn additional_headers(headers: &[(String, String)]) -> Map<String, Value> {
    let lookup = |name: &str| {
        headers
            .iter()
            .rev()
            .find(|(held, _)| held == name)
            .map(|(_, value)| value.as_str())
    };
    let header_names: Vec<String> = TYPED_KEYS.iter().map(|key| key.replace('_', "-")).collect();
    let typed = TYPED_KEYS
        .iter()
        .zip(&header_names)
        .filter_map(|(key, name)| Some(((*key).to_owned(), int_or_text(lookup(name)?))));
    let mut logged: Map<String, Value> = typed.collect();
    for (name, value) in headers {
        if !header_names.contains(&name.to_lowercase()) {
            logged.insert(name.clone(), Value::from(value.as_str()));
        }
    }
    logged
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn pairs(headers: &[(&str, &str)]) -> Vec<(String, String)> {
        headers
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }

    #[rstest]
    fn httpx_lowercases_names_and_joins_a_repeated_one_where_it_first_came() {
        assert_eq!(
            httpx_items(&pairs(&[("Via", "a"), ("x-id", "1"), ("via", "b")])),
            pairs(&[("via", "a, b"), ("x-id", "1")])
        );
    }

    #[rstest]
    fn anthropic_headers_are_prefixed_then_its_rate_limits_named_as_openai_names_them() {
        assert_eq!(
            anthropic_response_headers(&pairs(&[
                ("request-id", "req_1"),
                ("anthropic-ratelimit-requests-remaining", "99"),
            ])),
            pairs(&[
                ("llm_provider-request-id", "req_1"),
                ("llm_provider-anthropic-ratelimit-requests-remaining", "99"),
                ("x-ratelimit-remaining-requests", "99"),
            ])
        );
    }

    #[rstest]
    #[case::typed_first_as_ints(
        &[("llm_provider-request-id", "req_1"), ("x-ratelimit-remaining-requests", "99")],
        json!({"x_ratelimit_remaining_requests": 99, "llm_provider-request-id": "req_1"})
    )]
    #[case::a_typed_value_int_does_not_take(
        &[("x-ratelimit-reset-tokens", "1s")],
        json!({"x_ratelimit_reset_tokens": "1s"})
    )]
    #[case::other_headers_as_they_are(
        &[("retry-after", "60"), ("rate_limited_by", "api_key")],
        json!({"retry-after": "60", "rate_limited_by": "api_key"})
    )]
    #[case::none(&[], json!({}))]
    fn additional_headers_are_cleaned_as_python_cleans_them(
        #[case] headers: &[(&str, &str)],
        #[case] logged: Value,
    ) {
        assert_eq!(Value::Object(additional_headers(&pairs(headers))), logged);
    }
}
