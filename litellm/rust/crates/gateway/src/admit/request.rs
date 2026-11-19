//! What one request carries that the Python proxy reads and the native `/v1/messages` may not.
//!
//! Each [`Trait`] is found from the request alone. Whether it leaves the request to Python is the
//! plan's to say, since a config can make the native app read one as Python does.

use axum::http::{
    HeaderMap,
    header::{CONTENT_ENCODING, CONTENT_TYPE, USER_AGENT},
    request::Parts,
};
use litellm_llms_types::formats::messages::MessagesOptionalParams;
use serde_json::{Map, Value};

use super::Refused;

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub(super) enum Trait {
    /// An `x-litellm-*` header, which carries a request's model, tags, timeouts, retries, agent
    /// and session to the proxy.
    ProxyHeader,
    ContentEncoding,
    /// A content type other than JSON, since the proxy parses a form or protobuf body its own way.
    ContentType,
    /// A query parameter other than `beta`, as `api-version` is.
    Query,
    /// A Claude Code or Codex user agent, for which the proxy turns `drop_params` on.
    CodingAgent,
    /// A top-level key the Messages request type does not name, which the native route would send
    /// on to the provider. The proxy's own parameters (`user`, `tags`, `timeout`, `fallbacks`) are
    /// such.
    UnreadKey,
    CacheControl,
    Safeguards,
    /// The proxy checks who owns the thread and records it.
    Thread,
    /// The proxy resolves `container.skills` and can run them.
    Skills,
    /// `metadata` in a shape the proxy rewrites before the provider sees it.
    Metadata,
    /// A tool the proxy answers itself: its MCP gateway or the advisor.
    HostedTool,
}

impl Trait {
    /// Whether the native app reads this as the proxy does unless a finding says otherwise.
    pub(super) fn is_read(self) -> bool {
        matches!(self, Self::CodingAgent)
    }

    pub(super) fn part(self) -> Refused {
        match self {
            Self::ProxyHeader
            | Self::ContentEncoding
            | Self::ContentType
            | Self::Query
            | Self::CodingAgent => Refused::Headers,
            Self::UnreadKey
            | Self::CacheControl
            | Self::Safeguards
            | Self::Thread
            | Self::Skills
            | Self::Metadata
            | Self::HostedTool => Refused::Body,
        }
    }
}

const CALL_ID: &str = "x-litellm-call-id";
const CLAUDE_CODE: [&str; 2] = ["claude-cli/", "claude-code/"];
const HOSTED_TOOLS: [&str; 2] = ["mcp", "advisor_20260301"];

fn text(headers: &HeaderMap, name: impl axum::http::header::AsHeaderName) -> Option<&str> {
    headers.get(name)?.to_str().ok()
}

fn is_json(content_type: &str) -> bool {
    content_type
        .split(';')
        .next()
        .is_some_and(|essence| essence.trim().eq_ignore_ascii_case("application/json"))
}

/// The proxy's `is_claude_code_user_agent` and `is_codex_user_agent`.
fn is_coding_agent(user_agent: &str) -> bool {
    let codex = user_agent.split_at_checked(5).is_some_and(|(name, rest)| {
        name.eq_ignore_ascii_case("codex") && rest.starts_with(['-', '_', ' ', '/'])
    });
    codex
        || CLAUDE_CODE
            .iter()
            .any(|prefix| user_agent.starts_with(prefix))
}

fn carried<const N: usize>(found: [(Trait, bool); N]) -> Vec<Trait> {
    found
        .into_iter()
        .filter_map(|(carried, found)| found.then_some(carried))
        .collect()
}

/// The traits `request` carries outside its body.
pub(super) fn head(request: &Parts) -> Vec<Trait> {
    let headers = &request.headers;
    let query = request.uri.query().unwrap_or_default();
    carried([
        (
            Trait::ProxyHeader,
            headers
                .keys()
                .map(|name| name.as_str())
                .any(|name| name.starts_with("x-litellm-") && name != CALL_ID),
        ),
        (
            Trait::ContentEncoding,
            headers.contains_key(CONTENT_ENCODING),
        ),
        (
            Trait::ContentType,
            text(headers, CONTENT_TYPE).is_some_and(|content_type| !is_json(content_type)),
        ),
        (
            Trait::Query,
            query
                .split('&')
                .filter(|pair| !pair.is_empty())
                .any(|pair| pair.split('=').next() != Some("beta")),
        ),
        (
            Trait::CodingAgent,
            text(headers, USER_AGENT).is_some_and(is_coding_agent),
        ),
    ])
}

/// What a Messages request's body asks for.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Asked {
    pub model: Box<str>,
    pub carried: Vec<Trait>,
}

fn names_a_hosted_tool(fields: &Map<String, Value>) -> bool {
    fields
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|tool| tool.get("type")?.as_str())
        .any(|kind| HOSTED_TOOLS.contains(&kind))
}

/// The one shape the proxy and the native route both send on as written.
fn is_a_user_id(metadata: &Value) -> bool {
    metadata.as_object().is_some_and(|fields| {
        fields
            .iter()
            .all(|(key, value)| key == "user_id" && value.is_string())
    })
}

const REQUIRED: [&str; 2] = ["model", "messages"];

/// What the body `fields` ask for, read off its keys with no second pass over its values. `None`
/// when it names no model. Whether each value has the type its key takes is left to the route.
pub(super) fn body(fields: &Map<String, Value>) -> Option<Asked> {
    let sent = |key: &str| fields.get(key).filter(|value| !value.is_null());
    let is_read = |key: &str| {
        REQUIRED.contains(&key) || MessagesOptionalParams::fields().any(|name| name == key)
    };
    Some(Asked {
        model: fields.get("model")?.as_str()?.into(),
        carried: carried([
            (Trait::UnreadKey, fields.keys().any(|key| !is_read(key))),
            (Trait::CacheControl, sent("cache_control").is_some()),
            (Trait::Safeguards, sent("safeguards").is_some()),
            (Trait::Thread, sent("thread").is_some()),
            (
                Trait::Skills,
                sent("container").is_some_and(|container| container.get("skills").is_some()),
            ),
            (
                Trait::Metadata,
                sent("metadata").is_some_and(|metadata| !is_a_user_id(metadata)),
            ),
            (Trait::HostedTool, names_a_hosted_tool(fields)),
        ]),
    })
}

#[cfg(test)]
mod tests {
    use axum::http::Request;
    use rstest::rstest;

    use super::*;

    fn parts(uri: &str, header: Option<(&str, &str)>) -> Parts {
        header
            .into_iter()
            .fold(Request::post(uri), |request, (name, value)| {
                request.header(name, value)
            })
            .body(())
            .unwrap()
            .into_parts()
            .0
    }

    #[rstest]
    #[case::the_call_id(Some(("x-litellm-call-id", "abc")))]
    #[case::json(Some(("content-type", "application/json")))]
    #[case::json_with_a_charset(Some(("content-type", "Application/JSON; charset=utf-8")))]
    #[case::another_client(Some(("user-agent", "curl/8.7.1")))]
    #[case::a_session_of_another_client(Some(("x-claude-code-session-id", "0af404e1")))]
    #[case::no_header(None)]
    fn a_request_with_nothing_only_the_proxy_reads_carries_no_trait(
        #[case] header: Option<(&str, &str)>,
    ) {
        assert_eq!(head(&parts("/v1/messages?beta=true", header)), []);
    }

    #[rstest]
    #[case::a_proxy_header(("x-litellm-timeout", "5"), Trait::ProxyHeader)]
    #[case::an_encoded_body(("content-encoding", "gzip"), Trait::ContentEncoding)]
    #[case::a_form(("content-type", "application/x-www-form-urlencoded"), Trait::ContentType)]
    #[case::claude_code(("user-agent", "claude-cli/2.1.0 (external, cli)"), Trait::CodingAgent)]
    #[case::claude_codes_own_fetch(("user-agent", "claude-code/2.1.0"), Trait::CodingAgent)]
    #[case::codex(("user-agent", "Codex_cli_rs/0.40.0"), Trait::CodingAgent)]
    fn a_header_the_proxy_reads_is_a_trait(#[case] header: (&str, &str), #[case] carried: Trait) {
        assert_eq!(head(&parts("/v1/messages", Some(header))), [carried]);
    }

    #[rstest]
    #[case::a_name_that_only_starts_like_codex("codexa/1.0")]
    #[case::codex_alone("codex")]
    #[case::claude_code_later_in_the_value("agent claude-cli/2.1.0")]
    fn a_user_agent_the_proxy_does_not_match_is_no_coding_agent(#[case] user_agent: &str) {
        assert_eq!(
            head(&parts("/v1/messages", Some(("user-agent", user_agent)))),
            []
        );
    }

    #[rstest]
    #[case::another_parameter("/v1/messages?api-version=2024-06-01")]
    #[case::beside_beta("/v1/messages?beta=true&api-version=2024-06-01")]
    fn a_query_parameter_other_than_beta_is_a_trait(#[case] uri: &str) {
        assert_eq!(head(&parts(uri, None)), [Trait::Query]);
    }

    #[rstest]
    fn each_trait_a_request_carries_is_found() {
        let request = Request::post("/v1/messages")
            .header("x-litellm-tags", "a")
            .header("content-encoding", "zstd")
            .body(())
            .unwrap()
            .into_parts()
            .0;

        assert_eq!(head(&request), [Trait::ProxyHeader, Trait::ContentEncoding]);
    }

    fn fields(body: &str) -> Map<String, Value> {
        serde_json::from_str(body).unwrap()
    }

    fn asked(rest: &str) -> Option<Asked> {
        body(&fields(&format!(
            r#"{{"model":"claude","messages":[]{rest}}}"#
        )))
    }

    #[rstest]
    #[case::required_only("")]
    #[case::messages_params(r#","max_tokens":16,"stream":true,"system":"be brief""#)]
    #[case::a_user_id(r#","metadata":{"user_id":"user_1_account__session_2"}"#)]
    #[case::empty_metadata(r#","metadata":{}"#)]
    #[case::a_tool_the_provider_runs(
        r#","tools":[{"name":"get_weather","input_schema":{"type":"object"}},{"type":"web_search_20250305","name":"web_search"}]"#
    )]
    #[case::a_container_by_id(r#","container":"container_1""#)]
    #[case::a_param_sent_as_null(r#","thread":null,"metadata":null"#)]
    fn a_body_of_messages_params_carries_no_trait(#[case] rest: &str) {
        assert_eq!(
            asked(rest),
            Some(Asked {
                model: "claude".into(),
                carried: vec![]
            })
        );
    }

    #[rstest]
    #[case::a_proxy_param(r#","user":"end-user""#, Trait::UnreadKey)]
    #[case::an_unknown_param(r#","not_a_param":1"#, Trait::UnreadKey)]
    #[case::cache_control(r#","cache_control":{"type":"ephemeral"}"#, Trait::CacheControl)]
    #[case::safeguards(r#","safeguards":[{"type":"dangerous_tool_use"}]"#, Trait::Safeguards)]
    #[case::thread(r#","thread":{"type":"create"}"#, Trait::Thread)]
    #[case::skills(
        r#","container":{"skills":[{"skill_id":"litellm_skill_1"}]}"#,
        Trait::Skills
    )]
    #[case::a_user_id_the_proxy_rewrites(
        r#","metadata":{"user_id":{"session_id":"abc"}}"#,
        Trait::Metadata
    )]
    #[case::metadata_beyond_the_user_id(
        r#","metadata":{"user_id":"u","trace":"t"}"#,
        Trait::Metadata
    )]
    #[case::metadata_as_text(r#","metadata":"{\"user_id\":\"u\"}""#, Trait::Metadata)]
    #[case::an_mcp_gateway_tool(
        r#","tools":[{"type":"mcp","server_url":"litellm_proxy/mcp/github"}]"#,
        Trait::HostedTool
    )]
    #[case::the_advisor(
        r#","tools":[{"type":"advisor_20260301","name":"advisor","model":"claude-opus-5-5"}]"#,
        Trait::HostedTool
    )]
    fn a_body_with_something_the_proxy_reads_carries_its_trait(
        #[case] rest: &str,
        #[case] carried: Trait,
    ) {
        assert_eq!(
            asked(rest),
            Some(Asked {
                model: "claude".into(),
                carried: vec![carried]
            })
        );
    }

    #[rstest]
    #[case::no_model(r#"{"messages":[]}"#)]
    #[case::a_model_that_is_not_a_name(r#"{"model":7,"messages":[]}"#)]
    fn a_body_that_names_no_model_asks_for_nothing(#[case] sent: &str) {
        assert_eq!(body(&fields(sent)), None);
    }
}
