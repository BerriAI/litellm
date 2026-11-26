//! The `litellm_trace_id` the proxy gives a request from its headers, which the spend log row
//! names as its session: the call chain the caller names (`get_chain_id_from_headers`), else the
//! trace of a W3C `traceparent`, else a fresh id. `add_litellm_data_to_request` sets it so, and
//! a request whose body names a trace or session of its own is one the gateway's `admit` leaves
//! to Python.

use std::sync::LazyLock;

use axum::http::{HeaderMap, header::USER_AGENT};
use regex::Regex;

const EXPLICIT: [&str; 2] = ["x-litellm-trace-id", "x-litellm-session-id"];
/// Codex's own session headers, newest first, read only from Codex.
const CODEX: [&str; 4] = ["session-id", "session_id", "thread-id", "conversation_id"];
const BARE: &str = "x-session-id";
const TRACEPARENT: &str = "traceparent";
/// The longest trace or session id a header may name. Clients send UUIDs (36) and a traceparent
/// trace is 32 hex, so this leaves room for prefixed ids while keeping an oversized header out
/// of the spend log row. A longer value names no session, and the next header is read.
const MAX_TRACE_ID: usize = 256;

static SESSION_HEADER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^x-.+-session-id$").expect("session header pattern compiles"));
static SESSION_VALUE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-zA-Z0-9_\-]{8,}$").expect("session value pattern compiles"));
static CODEX_CLIENT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^codex[-_ /]").expect("codex pattern compiles"));
static TRACE_CONTEXT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^[0-9a-f]{2}-([0-9a-f]{32})-[0-9a-f]{16}-[0-9a-f]{2}$")
        .expect("traceparent pattern compiles")
});

fn text<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get(name)?
        .to_str()
        .ok()
        .filter(|value| !value.is_empty())
}

fn within(value: &str) -> bool {
    value.len() <= MAX_TRACE_ID
}

fn session(value: &str) -> bool {
    within(value) && SESSION_VALUE.is_match(value)
}

/// The call chain the caller named: an explicit LiteLLM header, any `x-<vendor>-session-id`,
/// Codex's own session header, then a bare `x-session-id`.
fn chain(headers: &HeaderMap) -> Option<&str> {
    let explicit = EXPLICIT
        .iter()
        .find_map(|name| text(headers, name).filter(|value| within(value)));
    let vendor = || {
        headers.iter().find_map(|(name, value)| {
            let name = name.as_str();
            let value = value.to_str().ok()?;
            (!EXPLICIT.contains(&name) && SESSION_HEADER.is_match(name) && session(value))
                .then_some(value)
        })
    };
    let codex = || {
        text(headers, USER_AGENT.as_str())
            .filter(|agent| CODEX_CLIENT.is_match(agent))
            .and_then(|_| {
                CODEX
                    .iter()
                    .find_map(|name| text(headers, name).filter(|value| session(value)))
            })
    };
    let bare = || text(headers, BARE).filter(|value| session(value));
    explicit.or_else(vendor).or_else(codex).or_else(bare)
}

/// The trace a `traceparent` carries, lowercased, unless it is the all-zero trace the spec
/// calls invalid.
fn traceparent(headers: &HeaderMap) -> Option<String> {
    let value = text(headers, TRACEPARENT)?.trim();
    let trace = TRACE_CONTEXT
        .captures(value)?
        .get(1)?
        .as_str()
        .to_lowercase();
    trace.bytes().any(|digit| digit != b'0').then_some(trace)
}

/// The request's `litellm_trace_id`.
pub(crate) fn of(headers: &HeaderMap) -> String {
    chain(headers)
        .map(str::to_owned)
        .or_else(|| traceparent(headers))
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string())
}

#[cfg(test)]
mod tests {
    use axum::http::{HeaderName, HeaderValue};
    use rstest::rstest;

    use super::*;

    const SESSION: &str = "e96634a3-fa28-4083-b354-55542e2dca01";
    const TRACE: &str = "4bf92f3577b34da6a3ce929d0e0e4736";

    fn headers(sent: &[(&str, &str)]) -> HeaderMap {
        sent.iter()
            .map(|(name, value)| {
                (
                    HeaderName::from_bytes(name.as_bytes()).unwrap(),
                    HeaderValue::from_str(value).unwrap(),
                )
            })
            .collect()
    }

    #[rstest]
    #[case::claude_codes_session(&[("x-claude-code-session-id", SESSION)], SESSION)]
    #[case::the_litellm_trace_over_a_vendor_session(
        &[("x-claude-code-session-id", SESSION), ("x-litellm-trace-id", "trace-1")],
        "trace-1"
    )]
    #[case::the_litellm_session(&[("x-litellm-session-id", "s")], "s")]
    #[case::codexs_session(
        &[("user-agent", "codex_cli_rs/0.40.0"), ("thread-id", SESSION), ("session-id", "session-123")],
        "session-123"
    )]
    #[case::a_bare_session(&[("x-session-id", SESSION)], SESSION)]
    #[case::a_vendor_session_over_a_bare_one(
        &[("x-session-id", "bare-session"), ("x-parent-session-id", SESSION)],
        SESSION
    )]
    #[case::a_traceparent(
        &[("traceparent", "00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01")],
        TRACE
    )]
    #[case::a_session_over_a_traceparent(
        &[("traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"), ("x-claude-code-session-id", SESSION)],
        SESSION
    )]
    fn the_trace_is_the_session_or_trace_the_caller_sent(
        #[case] sent: &[(&str, &str)],
        #[case] trace: &str,
    ) {
        assert_eq!(of(&headers(sent)), trace);
    }

    #[rstest]
    #[case::nothing(&[])]
    #[case::a_session_too_short(&[("x-claude-code-session-id", "abc")])]
    #[case::a_session_with_other_characters(&[("x-claude-code-session-id", "abc.def.ghi")])]
    #[case::codexs_header_from_another_client(&[("user-agent", "curl/8"), ("session-id", SESSION)])]
    #[case::an_all_zero_traceparent(
        &[("traceparent", "00-00000000000000000000000000000000-00f067aa0ba902b7-01")]
    )]
    #[case::a_malformed_traceparent(&[("traceparent", "00-4bf92f-00f067aa0ba902b7-01")])]
    fn a_request_that_names_no_trace_gets_a_fresh_one(#[case] sent: &[(&str, &str)]) {
        let headers = headers(sent);
        let (first, second) = (of(&headers), of(&headers));
        assert!(
            first != second && uuid::Uuid::parse_str(&first).is_ok(),
            "{first} {second}"
        );
    }

    #[rstest]
    fn a_trace_id_over_the_cap_names_no_session() {
        let (at_cap, over) = ("s".repeat(MAX_TRACE_ID), "s".repeat(MAX_TRACE_ID + 1));
        let cases = [
            (
                vec![("x-claude-code-session-id", at_cap.as_str())],
                Some(at_cap.as_str()),
            ),
            (
                vec![
                    ("x-litellm-trace-id", over.as_str()),
                    ("x-claude-code-session-id", SESSION),
                ],
                Some(SESSION),
            ),
            (
                vec![
                    ("x-claude-code-session-id", over.as_str()),
                    ("x-session-id", SESSION),
                ],
                Some(SESSION),
            ),
            (vec![("x-litellm-trace-id", over.as_str())], None),
        ];
        for (sent, named) in cases {
            let trace = of(&headers(&sent));
            match named {
                Some(named) => assert_eq!(trace, named),
                None => assert!(uuid::Uuid::parse_str(&trace).is_ok(), "{trace}"),
            }
        }
    }
}
