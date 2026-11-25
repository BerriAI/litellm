//! What the payload reads of the request's own headers: its `User-Agent` tags, the custom `x-`
//! headers the enterprise build logs, and whether the caller forwarded an Anthropic OAuth token.
//! Each takes the headers as the client sent them, by lowercase name in the order they came, and
//! mirrors the Python helper named on it, for a request the gateway serves: one with no
//! `x-litellm-*` header but `x-litellm-call-id`, and no setting that changes which headers the
//! proxy keeps (`litellm_key_header_name`, `forward_llm_provider_auth_headers`, MCP servers).

use std::collections::BTreeMap;

const USER_AGENT: &str = "user-agent";
const AUTHORIZATION: &str = "authorization";
const LITELLM_API_KEY: &str = "x-litellm-api-key";
/// `ANTHROPIC_OAUTH_TOKEN_PREFIX` as a bearer, which `anthropic_oauth_token_from_header` reads.
const ANTHROPIC_OAUTH_BEARER: &str = "Bearer sk-ant-oat";
/// The `x-` names of `SpecialHeaders`, which `clean_headers` drops from the logged copy.
const SPECIAL_HEADERS: [&str; 6] = [
    "x-api-key",
    "x-goog-api-key",
    "x-litellm-api-key",
    "x-mcp-auth",
    "x-mcp-servers",
    "x-mcp-access-groups",
];
/// `_MCP_SERVER_AUTH_HEADER_PREFIX`: `upstream_credential_headers` drops every header under it
/// that is not one of `SPECIAL_HEADERS`.
const MCP_SERVER_AUTH_HEADER_PREFIX: &str = "x-mcp-";
/// `_CREDENTIAL_HEADER_NAMES`, whose values `redact_credential_headers` masks, also under an
/// `x-pass-` prefix.
const CREDENTIAL_HEADER_NAMES: [&str; 9] = [
    "authorization",
    "api-key",
    "x-api-key",
    "x-goog-api-key",
    "ocp-apim-subscription-key",
    "x-litellm-api-key",
    "cookie",
    "proxy-authorization",
    "x-amz-security-token",
];
const PASS_PREFIX: &str = "x-pass-";
/// `REDACTED_HEADER_VALUE`.
const REDACTED_HEADER_VALUE: &str = "***REDACTED***";

/// The value a Python dict built from `headers` holds under `name`: the last one sent.
fn last<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .rev()
        .find(|(sent, _)| sent == name)
        .map(|(_, value)| value.as_str())
}

/// `StandardLoggingPayloadSetup._get_user_agent_tags(proxy_server_request)` with
/// `litellm.disable_add_user_agent_to_request_tags` off, the only value the gateway serves. The
/// logged headers are a dict, so a repeated `User-Agent` is read by its last value. These are the
/// only `request_tags` a served request has: body metadata tags and `extra_spend_tag_headers`
/// are left to Python.
pub fn user_agent_tags(headers: &[(String, String)]) -> Vec<String> {
    let Some(user_agent) = last(headers, USER_AGENT) else {
        return Vec::new();
    };
    let product = user_agent
        .contains('/')
        .then(|| user_agent.split('/').next().unwrap_or_default());
    product
        .into_iter()
        .chain([user_agent])
        .map(|tag| format!("User-Agent: {tag}"))
        .collect()
}

/// The `x-` headers the proxy keeps in `proxy_server_request["headers"]`, as `clean_headers`,
/// `upstream_credential_headers` and `redact_credential_headers` leave them.
fn logged(name: &str) -> bool {
    name.starts_with("x-")
        && !SPECIAL_HEADERS.contains(&name)
        && !name.starts_with(MCP_SERVER_AUTH_HEADER_PREFIX)
}

/// `_is_credential_header(name)`.
fn is_credential_header(name: &str) -> bool {
    CREDENTIAL_HEADER_NAMES.contains(&name.strip_prefix(PASS_PREFIX).unwrap_or(name))
}

/// `requester_custom_headers` as the enterprise
/// `StandardLoggingPayloadSetup.apply_enterprise_specific_metadata` sets it: every `x-` header of
/// the logged request headers, with a credential's value masked. `None` for a request with no
/// headers at all, whose metadata keeps the default.
pub fn requester_custom_headers(headers: &[(String, String)]) -> Option<BTreeMap<String, String>> {
    if headers.is_empty() {
        return None;
    }
    Some(
        headers
            .iter()
            .filter(|(name, _)| logged(name))
            .map(|(name, value)| {
                let value = if is_credential_header(name) {
                    REDACTED_HEADER_VALUE
                } else {
                    value
                };
                (name.clone(), value.to_owned())
            })
            .collect(),
    )
}

/// The `used_client_oauth_token` the proxy stamps on the request's metadata
/// (`add_provider_specific_headers_to_request`): whether the `Authorization` that
/// `clean_headers` keeps carries an Anthropic OAuth bearer. It keeps one only when the caller
/// authenticated with `x-litellm-api-key` (`litellm_auth_header_name`), which a request the
/// gateway serves never sends, so for those it is always false.
pub fn client_sent_oauth_token(headers: &[(String, String)]) -> bool {
    let authenticated_with_authorization = !headers.iter().any(|(name, _)| name == LITELLM_API_KEY)
        && headers.iter().any(|(name, _)| name == AUTHORIZATION);
    !authenticated_with_authorization
        && last(headers, AUTHORIZATION)
            .is_some_and(|value| value.starts_with(ANTHROPIC_OAUTH_BEARER))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn headers(sent: &[(&str, &str)]) -> Vec<(String, String)> {
        sent.iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }

    #[rstest]
    #[case::product_and_version(&[("user-agent", "claude-cli/2.1.0 (external, cli)")], &["User-Agent: claude-cli", "User-Agent: claude-cli/2.1.0 (external, cli)"])]
    #[case::no_slash(&[("user-agent", "curl")], &["User-Agent: curl"])]
    #[case::empty(&[("user-agent", "")], &["User-Agent: "])]
    #[case::leading_slash(&[("user-agent", "/x")], &["User-Agent: ", "User-Agent: /x"])]
    #[case::the_last_of_two(&[("user-agent", "a/1"), ("user-agent", "b")], &["User-Agent: b"])]
    #[case::none(&[("host", "proxy.test")], &[])]
    fn user_agent_tags_are_the_product_then_the_whole_value(
        #[case] sent: &[(&str, &str)],
        #[case] tags: &[&str],
    ) {
        assert_eq!(user_agent_tags(&headers(sent)), tags);
    }

    #[rstest]
    fn custom_headers_are_the_x_headers_the_proxy_logs() {
        let sent = headers(&[
            ("host", "proxy.test"),
            ("x-litellm-call-id", "call-1"),
            ("x-api-key", "sk-credential-value"),
            ("x-goog-api-key", "credential-value"),
            ("x-mcp-github-authorization", "credential-value"),
            ("x-pass-authorization", "Bearer credential-value"),
            ("x-amz-security-token", "credential-value"),
            ("x-team", "first"),
            ("x-team", "second"),
            ("anthropic-beta", "a"),
        ]);
        assert_eq!(
            requester_custom_headers(&sent),
            Some(BTreeMap::from_iter(
                [
                    ("x-litellm-call-id", "call-1"),
                    ("x-pass-authorization", REDACTED_HEADER_VALUE),
                    ("x-amz-security-token", REDACTED_HEADER_VALUE),
                    ("x-team", "second"),
                ]
                .map(|(name, value)| (name.to_owned(), value.to_owned()))
            ))
        );
    }

    #[rstest]
    #[case::no_headers(&[], None)]
    #[case::no_x_header(&[("host", "proxy.test")], Some(BTreeMap::new()))]
    fn custom_headers_are_empty_only_with_headers(
        #[case] sent: &[(&str, &str)],
        #[case] custom: Option<BTreeMap<String, String>>,
    ) {
        assert_eq!(requester_custom_headers(&headers(sent)), custom);
    }

    #[rstest]
    #[case::authenticated_with_the_bearer(&[("authorization", "Bearer sk-ant-oat01-x"), ("x-api-key", "sk-1")], false)]
    #[case::beside_a_litellm_key_header(&[("x-litellm-api-key", "sk-1"), ("authorization", "Bearer sk-ant-oat01-x")], true)]
    #[case::not_a_bearer(&[("x-litellm-api-key", "sk-1"), ("authorization", "sk-ant-oat01-x")], false)]
    #[case::an_api_key(&[("x-litellm-api-key", "sk-1"), ("authorization", "Bearer sk-ant-api03-x")], false)]
    #[case::no_authorization(&[("x-api-key", "sk-ant-oat01-x")], false)]
    fn an_oauth_token_counts_only_where_the_proxy_keeps_it(
        #[case] sent: &[(&str, &str)],
        #[case] stamped: bool,
    ) {
        assert_eq!(client_sent_oauth_token(&headers(sent)), stamped);
    }
}
