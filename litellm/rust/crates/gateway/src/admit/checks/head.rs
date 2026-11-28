//! What a request is and carries outside its body.

use axum::http::{
    HeaderMap, Method,
    header::{CONTENT_ENCODING, CONTENT_TYPE},
    request::Parts,
};

use super::{Admit, Reason, Refused, config::messages_only};

const MESSAGES: &str = "/v1/messages";
const CALL_ID: &str = "x-litellm-call-id";
const FORWARDED_FOR: &str = "x-forwarded-for";

/// The native app serves `POST /v1/messages` and nothing else, and not even that when a setting
/// Python reads only there is on.
pub(super) fn route(admit: &Admit, request: &Parts) -> Option<Reason> {
    if request.method != Method::POST || request.uri.path() != MESSAGES {
        return Some(Reason::new(Refused::Route, "unserved_route"));
    }
    messages_only(&admit.config).map(|over| Reason::new(Refused::Settings, over))
}

/// Under `allowed_ips`, the address Python checks is the one its server resolved, which reads
/// `x-forwarded-for` from a peer the operator trusts. Only `use_x_forwarded_for` makes the proxy
/// read the header itself, so with it off a request that sends one is left to Python.
pub(super) fn forwarded_for(admit: &Admit, request: &Parts) -> Option<Reason> {
    (admit.allowlists.client_addresses.reads_address()
        && !admit.config.general_settings.use_x_forwarded_for
        && request.headers.contains_key(FORWARDED_FOR))
    .then(|| Reason::new(Refused::Headers, "general_settings.allowed_ips"))
}

pub(super) fn headers(_: &Admit, request: &Parts) -> Option<Reason> {
    let headers = &request.headers;
    let query = request.uri.query().unwrap_or_default();
    [
        (
            "proxy_header",
            headers
                .keys()
                .map(|name| name.as_str())
                .any(|name| name.starts_with("x-litellm-") && name != CALL_ID),
        ),
        ("content_encoding", headers.contains_key(CONTENT_ENCODING)),
        (
            "content_type",
            text(headers, CONTENT_TYPE).is_some_and(|content_type| !is_json(content_type)),
        ),
        (
            "query",
            query
                .split('&')
                .filter(|pair| !pair.is_empty())
                .any(|pair| pair.split('=').next() != Some("beta")),
        ),
    ]
    .into_iter()
    .find(|(_, found)| *found)
    .map(|(over, _)| Reason::new(Refused::Headers, over))
}

fn text(headers: &HeaderMap, name: impl axum::http::header::AsHeaderName) -> Option<&str> {
    headers.get(name)?.to_str().ok()
}

fn is_json(content_type: &str) -> bool {
    content_type
        .split(';')
        .next()
        .is_some_and(|essence| essence.trim().eq_ignore_ascii_case("application/json"))
}

#[cfg(test)]
mod tests {
    use axum::http::Request;
    use litellm_config::Config;
    use rstest::rstest;

    use super::*;

    const DROP_PARAMS: &str = "litellm_settings:\n  drop_params: true\n";

    fn head(yaml: &str, method: Method, uri: &str, sent: &[(&str, &str)]) -> Option<String> {
        let admit = Admit::over(&Config::from_yaml(yaml).unwrap(), vec![]);
        let request = sent
            .iter()
            .fold(
                Request::builder().method(method).uri(uri),
                |request, (name, value)| request.header(*name, *value),
            )
            .body(())
            .unwrap()
            .into_parts()
            .0;
        let checks: [fn(&Admit, &Parts) -> Option<Reason>; 2] = [route, headers];
        checks
            .iter()
            .find_map(|check| check(&admit, &request))
            .map(|reason| reason.over.into())
    }

    fn messages(yaml: &str, uri: &str, sent: &[(&str, &str)]) -> Option<String> {
        head(yaml, Method::POST, uri, sent)
    }

    #[rstest]
    #[case::a_longer_path(Method::POST, "/v1/messages/count_tokens")]
    #[case::another_method(Method::GET, "/v1/messages")]
    fn only_the_exact_route_is_served(#[case] method: Method, #[case] uri: &str) {
        assert_eq!(
            head(DROP_PARAMS, method, uri, &[]).as_deref(),
            Some("unserved_route")
        );
    }

    #[rstest]
    #[case::json_with_a_charset(&[("content-type", "Application/JSON; charset=utf-8")])]
    #[case::the_call_id(&[("x-litellm-call-id", "abc")])]
    #[case::a_session_of_another_client(&[("x-claude-code-session-id", "0af404e1")])]
    #[case::a_coding_agent(&[("user-agent", "claude-cli/2.1.0")])]
    fn a_header_only_the_client_reads_is_not_named(#[case] sent: &[(&str, &str)]) {
        assert_eq!(messages(DROP_PARAMS, "/v1/messages", sent), None);
    }

    #[rstest]
    fn a_form_body_is_named() {
        let form = [("content-type", "application/x-www-form-urlencoded")];

        assert_eq!(
            messages(DROP_PARAMS, "/v1/messages", &form).as_deref(),
            Some("content_type")
        );
    }

    #[rstest]
    #[case::beta_alone("/v1/messages?beta=true", None)]
    #[case::beside_beta("/v1/messages?beta=true&api-version=2024-06-01", Some("query"))]
    fn only_the_beta_query_parameter_is_served(#[case] uri: &str, #[case] over: Option<&str>) {
        assert_eq!(messages(DROP_PARAMS, uri, &[]).as_deref(), over);
    }
}
