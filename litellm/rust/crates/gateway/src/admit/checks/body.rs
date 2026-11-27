//! What a Messages request carries in its body.

use litellm_gateway_inference::messages::REQUEST_SETTINGS;
use litellm_llms_types::formats::messages::MessagesOptionalParams;
use litellm_python_compat::unicode::isspace;
use serde_json::{Map, Value};

use super::{Admit, Reason, Refused};

const REQUIRED: [&str; 2] = ["model", "messages"];
const HOSTED_TOOLS: [&str; 2] = ["mcp", "advisor_20260301"];

/// A key the Messages request type does not name is sent on to the provider by the native route,
/// and the proxy reads its own (`user`, `tags`, `timeout`, `fallbacks`). The native route reads
/// the request's messages settings (`drop_params`, `additional_drop_params`) as the proxy does.
/// The rest are Messages params the proxy acts on before the provider sees them.
pub(super) fn fields(_: &Admit, fields: &Map<String, Value>) -> Option<Reason> {
    let sent = |key: &str| fields.get(key).filter(|value| !value.is_null());
    let is_read = |key: &str| {
        REQUIRED.contains(&key)
            || REQUEST_SETTINGS.contains(&key)
            || MessagesOptionalParams::fields().any(|name| name == key)
    };
    [
        (
            "unreadable",
            !fields.get("model").is_some_and(Value::is_string),
        ),
        ("unread_key", fields.keys().any(|key| !is_read(key))),
        ("cache_control", sent("cache_control").is_some()),
        ("safeguards", sent("safeguards").is_some()),
        ("thread", sent("thread").is_some()),
        (
            "skills",
            sent("container").is_some_and(|container| container.get("skills").is_some()),
        ),
        (
            "metadata",
            sent("metadata").is_some_and(|metadata| !is_a_user_id(metadata)),
        ),
        ("hosted_tool", names_a_hosted_tool(fields)),
    ]
    .into_iter()
    .find(|(_, found)| *found)
    .map(|(over, _)| Reason::new(Refused::Body, over))
}

/// The proxy's MCP gateway and the advisor are tools the proxy answers itself.
fn names_a_hosted_tool(fields: &Map<String, Value>) -> bool {
    fields
        .get("tools")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|tool| tool.get("type")?.as_str())
        .any(|kind| HOSTED_TOOLS.contains(&kind))
}

/// The end user a body's `metadata.user_id` names (`get_end_user_id_from_request_body`), trimmed
/// of whitespace (`isspace`), and nobody when nothing is left. Every other place the proxy reads an
/// end user from is a key or a header an earlier check sends to Python.
pub(in crate::admit) fn end_user_id(fields: &Map<String, Value>) -> Option<&str> {
    let named = fields.get("metadata")?.get("user_id")?.as_str()?;
    Some(named.trim_matches(isspace)).filter(|end_user_id| !end_user_id.is_empty())
}

/// Any other `metadata` is in a shape the proxy rewrites before the provider sees it.
fn is_a_user_id(metadata: &Value) -> bool {
    metadata.as_object().is_some_and(|fields| {
        fields
            .iter()
            .all(|(key, value)| key == "user_id" && value.is_string())
    })
}

#[cfg(test)]
mod tests {
    use litellm_config::Config;
    use rstest::rstest;

    use super::*;

    fn named(body: &str) -> Option<String> {
        let admit = Admit::over(&Config::default(), vec![]);
        let sent: Map<String, Value> = serde_json::from_str(body).unwrap();
        fields(&admit, &sent).map(|reason| reason.over.into())
    }

    fn asking(rest: &str) -> Option<String> {
        named(&format!(r#"{{"model":"claude","messages":[]{rest}}}"#))
    }

    #[rstest]
    #[case::a_user_id(r#","metadata":{"user_id":"user_1_account__session_2"}"#)]
    #[case::empty_metadata(r#","metadata":{}"#)]
    #[case::a_tool_the_provider_runs(
        r#","tools":[{"name":"get_weather","input_schema":{"type":"object"}},{"type":"web_search_20250305","name":"web_search"}]"#
    )]
    #[case::a_container_by_id(r#","container":"container_1""#)]
    #[case::a_param_sent_as_null(r#","thread":null,"metadata":null"#)]
    #[case::the_requests_messages_settings(
        r#","drop_params":"yes","additional_drop_params":["metadata.user_id"]"#
    )]
    fn a_body_of_messages_params_is_kept(#[case] rest: &str) {
        assert_eq!(asking(rest), None);
    }

    #[rstest]
    #[case::an_unknown_param(r#","not_a_param":1"#, "unread_key")]
    #[case::cache_control(r#","cache_control":{"type":"ephemeral"}"#, "cache_control")]
    #[case::safeguards(r#","safeguards":[{"type":"dangerous_tool_use"}]"#, "safeguards")]
    #[case::thread(r#","thread":{"type":"create"}"#, "thread")]
    #[case::skills(
        r#","container":{"skills":[{"skill_id":"litellm_skill_1"}]}"#,
        "skills"
    )]
    #[case::a_user_id_the_proxy_rewrites(
        r#","metadata":{"user_id":{"session_id":"abc"}}"#,
        "metadata"
    )]
    #[case::metadata_beyond_the_user_id(r#","metadata":{"user_id":"u","trace":"t"}"#, "metadata")]
    #[case::metadata_as_text(r#","metadata":"{\"user_id\":\"u\"}""#, "metadata")]
    #[case::the_advisor(
        r#","tools":[{"type":"advisor_20260301","name":"advisor","model":"claude-opus-5-5"}]"#,
        "hosted_tool"
    )]
    fn a_body_with_something_the_proxy_reads_is_named(#[case] rest: &str, #[case] over: &str) {
        assert_eq!(asking(rest).as_deref(), Some(over));
    }

    #[rstest]
    #[case::text(r#"{"user_id":"eu-1"}"#, Some("eu-1"))]
    #[case::text_inside_whitespace(
        r#"{"user_id":" \u001c\u0085eu-1\u3000\n"}"#,
        Some("eu-1")
    )]
    #[case::space_alone(r#"{"user_id":" \u001f\t"}"#, None)]
    #[case::empty_text(r#"{"user_id":""}"#, None)]
    #[case::no_user_id("{}", None)]
    fn the_end_user_is_the_user_id_trimmed_of_whitespace(
        #[case] metadata: &str,
        #[case] end_user: Option<&str>,
    ) {
        let sent: Map<String, Value> =
            serde_json::from_str(&format!(r#"{{"metadata":{metadata}}}"#)).unwrap();
        assert_eq!(end_user_id(&sent), end_user);
    }

    #[rstest]
    #[case::no_model(r#"{"messages":[]}"#)]
    #[case::a_model_that_is_not_a_name(r#"{"model":7,"messages":[]}"#)]
    fn a_body_that_names_no_model_is_unreadable(#[case] sent: &str) {
        assert_eq!(named(sent).as_deref(), Some("unreadable"));
    }
}
