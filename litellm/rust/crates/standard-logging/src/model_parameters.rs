//! The payload's `model_parameters`: the call's optional params less the ones that carry the
//! prompt, with credentials in tool entries redacted, as
//! `ModelParamHelper.get_standard_logging_model_parameters`
//! (litellm/litellm_core_utils/model_param_helper.py) logs them.

use serde_json::{Map, Value};

/// The names `get_exclude_params_for_model_parameters` and `_get_exclude_kwargs` take out of
/// Python's `relevant_logging_args`.
const EXCLUDED: [&str; 6] = [
    "messages",
    "prompt",
    "input",
    "system",
    "metadata",
    "litellm_metadata",
];

/// `_PARAMS_WITH_CREDENTIAL_ENTRIES`: the params whose list entries may carry credentials.
const PARAMS_WITH_CREDENTIAL_ENTRIES: [&str; 2] = ["tools", "mcp_servers"];

/// `_ENTRY_CREDENTIAL_KEYS`: the keys of such an entry whose values are redacted.
const ENTRY_CREDENTIAL_KEYS: [&str; 3] = ["authorization", "authorization_token", "headers"];

/// `secret_redaction.REDACTED`.
const REDACTED: &str = "REDACTED";

/// `get_standard_logging_model_parameters` over the optional params of a Messages call. Python
/// keeps the keys in `relevant_logging_args`, every request param of the APIs LiteLLM serves
/// (the Anthropic Messages request's among them) less [`EXCLUDED`]. Every Messages optional param
/// is one of those, so for them the filter is that list alone. In each object entry of `tools`
/// and `mcp_servers` the credential keys are replaced by `REDACTED`.
pub fn messages_model_parameters(optional_params: Map<String, Value>) -> Map<String, Value> {
    optional_params
        .into_iter()
        .filter(|(name, _)| !EXCLUDED.contains(&name.as_str()))
        .map(|(name, value)| {
            if PARAMS_WITH_CREDENTIAL_ENTRIES.contains(&name.as_str()) {
                (name, without_entry_credentials(value))
            } else {
                (name, value)
            }
        })
        .collect()
}

/// `_without_entry_credentials`: a list whose object entries have their credential keys
/// redacted. Any other value is kept.
fn without_entry_credentials(value: Value) -> Value {
    match value {
        Value::Array(entries) => entries
            .into_iter()
            .map(|entry| match entry {
                Value::Object(fields) => Value::Object(
                    fields
                        .into_iter()
                        .map(|(key, value)| {
                            if ENTRY_CREDENTIAL_KEYS.contains(&key.as_str()) {
                                (key, Value::from(REDACTED))
                            } else {
                                (key, value)
                            }
                        })
                        .collect(),
                ),
                other => other,
            })
            .collect(),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Map, Value, json};

    use super::*;

    fn params(value: Value) -> Map<String, Value> {
        let Value::Object(fields) = value else {
            panic!("params are an object");
        };
        fields
    }

    #[rstest]
    #[case::sampling(
        json!({"max_tokens": 64, "temperature": 0.5, "stream": false}),
        json!({"max_tokens": 64, "temperature": 0.5, "stream": false})
    )]
    #[case::prompt_and_metadata(
        json!({"max_tokens": 64, "system": "be brief", "metadata": {"user_id": "u"}}),
        json!({"max_tokens": 64})
    )]
    #[case::tool_credentials(
        json!({"tools": [
            {"name": "t", "headers": {"x": "y"}, "authorization": "Bearer s"},
            "text"
        ]}),
        json!({"tools": [{"name": "t", "headers": "REDACTED", "authorization": "REDACTED"}, "text"]})
    )]
    #[case::mcp_server_token(
        json!({"mcp_servers": [{"url": "u", "authorization_token": "s"}]}),
        json!({"mcp_servers": [{"url": "u", "authorization_token": "REDACTED"}]})
    )]
    #[case::tools_that_are_no_list(json!({"tools": {"authorization": "s"}}), json!({"tools": {"authorization": "s"}}))]
    #[case::credentials_elsewhere(
        json!({"tool_choice": {"authorization": "s"}}),
        json!({"tool_choice": {"authorization": "s"}})
    )]
    fn the_prompt_params_are_left_out_and_tool_credentials_redacted(
        #[case] optional_params: Value,
        #[case] expected: Value,
    ) {
        assert_eq!(
            Value::Object(messages_model_parameters(params(optional_params))),
            expected
        );
    }
}
