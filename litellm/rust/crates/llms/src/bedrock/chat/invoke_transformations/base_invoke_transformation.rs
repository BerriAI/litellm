use litellm_python_compat::{json as python_json, repr};
use serde_json::{Map, Value};

use crate::Error;

pub const GUARDRAIL_CONFIG_PARAM: &str = "guardrailConfig";

pub(crate) const GUARDRAIL_IDENTIFIER_HEADER: &str = "X-Amzn-Bedrock-GuardrailIdentifier";
pub(crate) const GUARDRAIL_VERSION_HEADER: &str = "X-Amzn-Bedrock-GuardrailVersion";
const GUARDRAIL_TRACE_HEADER: &str = "X-Amzn-Bedrock-Trace";
const GUARDRAIL_CONFIG_EXPECTED_FORMAT: &str = "{'guardrailIdentifier': str, 'guardrailVersion': str, 'trace': 'enabled'|'disabled'|'enabled_full'}";
const GUARDRAIL_TRACE_VALUES: [&str; 3] = ["enabled", "disabled", "enabled_full"];

fn python_repr(value: &Value) -> String {
    repr::to_str(&python_json::from_json(value.clone()))
}

fn python_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(number) if number.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

fn validation_error(location: &str, message: &str, kind: &str, input: &Value) -> String {
    let location = if location.is_empty() {
        String::new()
    } else {
        format!("\n{location}")
    };
    format!(
        "1 validation error for GuardrailConfigBlock{location}\n  {message} [type={kind}, input_value={}, input_type={}]",
        python_repr(input),
        python_type_name(input)
    )
}

fn guardrail_violation(config: &Map<String, Value>) -> Option<String> {
    ["guardrailIdentifier", "guardrailVersion"]
        .into_iter()
        .find_map(|field| {
            config
                .get(field)
                .filter(|value| !value.is_string())
                .map(|value| {
                    validation_error(
                        field,
                        "Input should be a valid string",
                        "string_type",
                        value,
                    )
                })
        })
        .or_else(|| {
            config
                .get("trace")
                .filter(|value| {
                    !value
                        .as_str()
                        .is_some_and(|trace| GUARDRAIL_TRACE_VALUES.contains(&trace))
                })
                .map(|value| {
                    validation_error(
                        "trace",
                        "Input should be 'enabled', 'disabled' or 'enabled_full'",
                        "literal_error",
                        value,
                    )
                })
        })
}

fn guardrail_error(message: String) -> Error {
    Error::Http(litellm_http::Error::upstream(400, message))
}

/// Port of Python's `bedrock_invoke_guardrail_headers`.
pub(crate) fn bedrock_invoke_guardrail_headers(
    raw: &Value,
) -> Result<Vec<(String, String)>, Error> {
    let Value::Object(config) = raw else {
        return Err(guardrail_error(format!(
            "Invalid guardrailConfig={}. Expected format: {GUARDRAIL_CONFIG_EXPECTED_FORMAT}. Error: {}",
            python_repr(raw),
            validation_error("", "Input should be a valid dictionary", "dict_type", raw)
        )));
    };
    if let Some(violation) = guardrail_violation(config) {
        return Err(guardrail_error(format!(
            "Invalid guardrailConfig={}. Expected format: {GUARDRAIL_CONFIG_EXPECTED_FORMAT}. Error: {violation}",
            python_repr(raw)
        )));
    }
    let Some(identifier) = config.get("guardrailIdentifier").and_then(Value::as_str) else {
        return Err(guardrail_error(format!(
            "guardrailConfig={} is missing 'guardrailIdentifier'. Expected format: {GUARDRAIL_CONFIG_EXPECTED_FORMAT}",
            python_repr(raw)
        )));
    };
    Ok([
        (GUARDRAIL_IDENTIFIER_HEADER, Some(identifier.to_string())),
        (
            GUARDRAIL_VERSION_HEADER,
            config
                .get("guardrailVersion")
                .and_then(Value::as_str)
                .map(str::to_string),
        ),
        (
            GUARDRAIL_TRACE_HEADER,
            config
                .get("trace")
                .and_then(Value::as_str)
                .map(str::to_uppercase),
        ),
    ]
    .into_iter()
    .filter_map(|(name, value)| Some((name.to_string(), value?)))
    .collect())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn rejection(error: Error) -> (u16, String) {
        let Error::Http(error) = error else {
            panic!("expected a 400, got {error:?}");
        };
        let (status, body) = error.upstream_response().unwrap();
        (status, body.to_string())
    }

    #[rstest]
    #[case::enabled("enabled", "ENABLED")]
    #[case::disabled("disabled", "DISABLED")]
    #[case::enabled_full("enabled_full", "ENABLED_FULL")]
    fn guardrail_config_becomes_invoke_headers(#[case] trace: &str, #[case] expected: &str) {
        assert_eq!(
            bedrock_invoke_guardrail_headers(
                &json!({"guardrailIdentifier": "gid", "guardrailVersion": "2", "trace": trace})
            )
            .unwrap(),
            vec![
                (GUARDRAIL_IDENTIFIER_HEADER.to_string(), "gid".to_string()),
                (GUARDRAIL_VERSION_HEADER.to_string(), "2".to_string()),
                (GUARDRAIL_TRACE_HEADER.to_string(), expected.to_string()),
            ]
        );
    }

    #[test]
    fn a_guardrail_config_without_an_identifier_is_a_400() {
        assert_eq!(
            rejection(bedrock_invoke_guardrail_headers(&json!({"guardrailVersion": "2"})).unwrap_err()),
            (
                400,
                "guardrailConfig={'guardrailVersion': '2'} is missing 'guardrailIdentifier'. Expected format: {'guardrailIdentifier': str, 'guardrailVersion': str, 'trace': 'enabled'|'disabled'|'enabled_full'}".to_string()
            )
        );
    }

    #[rstest]
    #[case::bad_trace(json!({"guardrailIdentifier": "gid", "trace": "loud"}))]
    #[case::not_a_dict(json!("gid"))]
    #[case::non_string_version(json!({"guardrailIdentifier": "gid", "guardrailVersion": 2}))]
    fn an_invalid_guardrail_config_is_a_400(#[case] raw: Value) {
        let (status, body) = rejection(bedrock_invoke_guardrail_headers(&raw).unwrap_err());
        assert_eq!(status, 400);
        assert!(
            body.starts_with(&format!(
                "Invalid guardrailConfig={}. Expected format: ",
                python_repr(&raw)
            )),
            "{body}"
        );
    }
}
