//! What a request is and carries outside its body.

use axum::http::{
    HeaderMap, Method,
    header::{CONTENT_ENCODING, CONTENT_TYPE, USER_AGENT},
    request::Parts,
};
use litellm_config::LiteLlmSettings;

use super::{Admit, Reason, Refused, config::messages_only};

const MESSAGES: &str = "/v1/messages";
const CALL_ID: &str = "x-litellm-call-id";
const CLAUDE_CODE: [&str; 2] = ["claude-cli/", "claude-code/"];

/// The native app serves `POST /v1/messages` and nothing else, and not even that when a setting
/// Python reads only there is on.
pub(super) fn route(admit: &Admit, request: &Parts) -> Option<Reason> {
    if request.method != Method::POST || request.uri.path() != MESSAGES {
        return Some(Reason::new(Refused::Route, "unserved_route"));
    }
    messages_only(&admit.config).map(|over| Reason::new(Refused::Settings, over))
}

pub(super) fn headers(admit: &Admit, request: &Parts) -> Option<Reason> {
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
        (
            "litellm_settings.drop_params unset",
            text(headers, USER_AGENT).is_some_and(is_coding_agent)
                && defaults_differ_for_a_coding_agent(&admit.config.litellm_settings),
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

/// Claude Code or Codex, for which the proxy turns `drop_params` on.
fn is_coding_agent(user_agent: &str) -> bool {
    let codex = user_agent.split_at_checked(5).is_some_and(|(name, rest)| {
        name.eq_ignore_ascii_case("codex") && rest.starts_with(['-', '_', ' ', '/'])
    });
    codex
        || CLAUDE_CODE
            .iter()
            .any(|prefix| user_agent.starts_with(prefix))
}

/// The proxy turns `drop_params` on for a coding agent unless the setting is set, and the native
/// app only reads the setting.
fn defaults_differ_for_a_coding_agent(settings: &LiteLlmSettings) -> bool {
    settings.drop_params.is_none()
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

    #[rstest]
    #[case::claude_codes_own_fetch("claude-code/2.1.0", true)]
    #[case::codex_in_any_case("Codex_cli_rs/0.40.0", true)]
    #[case::a_name_that_only_starts_like_codex("codexa/1.0", false)]
    #[case::codex_alone("codex", false)]
    #[case::claude_code_later_in_the_value("agent claude-cli/2.1.0", false)]
    fn a_coding_agent_is_matched_as_the_proxy_matches_it(
        #[case] user_agent: &str,
        #[case] matched: bool,
    ) {
        assert_eq!(
            messages("", "/v1/messages", &[("user-agent", user_agent)]).is_some(),
            matched
        );
    }

    #[rstest]
    #[case::a_word_that_is_no_flag("litellm_settings:\n  drop_params: maybe\n", true)]
    #[case::text_that_reads_as_on("litellm_settings:\n  drop_params: \"yes\"\n", false)]
    #[case::off("litellm_settings:\n  drop_params: false\n", false)]
    fn drop_params_keeps_a_coding_agent_only_when_it_reads_as_a_flag(
        #[case] yaml: &str,
        #[case] named: bool,
    ) {
        let agent = [("user-agent", "claude-cli/2.1.0")];

        assert_eq!(messages(yaml, "/v1/messages", &agent).is_some(), named);
    }
}
