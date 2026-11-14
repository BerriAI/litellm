use litellm_python_compat::{json as python_json, repr};
use serde_json::{Map, Value};

use crate::{Error, base_llm::auth::Headers};

pub const GUARDRAIL_CONFIG_PARAM: &str = "guardrailConfig";
pub const REQUEST_METADATA_FIELDS_PARAM: &str = "bedrock_request_metadata_fields";
pub const METADATA_PARAMS: [&str; 2] = ["metadata", "litellm_metadata"];

const GUARDRAIL_IDENTIFIER_HEADER: &str = "X-Amzn-Bedrock-GuardrailIdentifier";
const GUARDRAIL_VERSION_HEADER: &str = "X-Amzn-Bedrock-GuardrailVersion";
const GUARDRAIL_TRACE_HEADER: &str = "X-Amzn-Bedrock-Trace";
const GUARDRAIL_CONFIG_EXPECTED_FORMAT: &str = "{'guardrailIdentifier': str, 'guardrailVersion': str, 'trace': 'enabled'|'disabled'|'enabled_full'}";
const GUARDRAIL_TRACE_VALUES: [&str; 3] = ["enabled", "disabled", "enabled_full"];

pub const REQUEST_METADATA_HEADER: &str = "X-Amzn-Bedrock-Request-Metadata";
const REQUEST_METADATA_MAX_PAIRS: usize = 16;
const REQUEST_METADATA_IDENTITY_PREFIX: &str = "user_api_key_";
const REQUEST_METADATA_CLIENT_FIELD: &str = "spend_logs_metadata";
const REQUEST_METADATA_HEADER_MAX_CHARS: usize = 8500;
const REQUEST_METADATA_MAX_CHARS: usize = 256;

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

pub(super) fn guardrail_headers(raw: &Value) -> Result<Vec<(String, String)>, Error> {
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

fn is_allowed_character(character: char) -> bool {
    character.is_ascii_alphanumeric()
        || matches!(
            character,
            ' ' | '\t'
                | '\n'
                | '\r'
                | '\x0b'
                | '\x0c'
                | ':'
                | '_'
                | '@'
                | '$'
                | '#'
                | '='
                | '/'
                | '+'
                | ','
                | '.'
                | '-'
        )
}

fn matches_constraint(text: &str, minimum: usize) -> bool {
    let count = text.chars().count();
    (minimum..=REQUEST_METADATA_MAX_CHARS).contains(&count)
        && text.chars().all(is_allowed_character)
}

fn is_forwardable(key: &str, value: &str) -> bool {
    matches_constraint(key, 1) && matches_constraint(value, 0)
}

fn is_reserved(key: &str) -> bool {
    key.trim()
        .to_lowercase()
        .starts_with(REQUEST_METADATA_IDENTITY_PREFIX)
}

fn text_pairs(source: Option<&Value>) -> Vec<(&str, &str)> {
    match source {
        Some(Value::Object(fields)) => fields
            .iter()
            .filter_map(|(key, value)| Some((key.as_str(), value.as_str()?)))
            .collect(),
        _ => Vec::new(),
    }
}

fn allowed_fields(configured: Option<&Value>) -> Vec<String> {
    let Some(Value::Array(fields)) = configured else {
        return Vec::new();
    };
    let fields: Vec<String> = fields
        .iter()
        .map(|field| match field {
            Value::String(text) => text.clone(),
            other => python_repr(other),
        })
        .collect();
    fields
        .iter()
        .enumerate()
        .filter(|(index, field)| !fields[..*index].contains(field))
        .map(|(_, field)| field.clone())
        .collect()
}

fn resolved_pairs(sources: &[&Map<String, Value>], allowed: &[String]) -> Vec<(String, String)> {
    let identity: Vec<(String, String)> = allowed
        .iter()
        .filter(|field| field.starts_with(REQUEST_METADATA_IDENTITY_PREFIX))
        .filter_map(|field| {
            let value = sources
                .iter()
                .find_map(|source| source.get(field.as_str()).and_then(Value::as_str))?;
            is_forwardable(field, value).then(|| (field.clone(), value.to_string()))
        })
        .take(REQUEST_METADATA_MAX_PAIRS)
        .collect();
    let spend_logs_pairs: Vec<(&str, &str)> = if allowed
        .iter()
        .any(|field| field == REQUEST_METADATA_CLIENT_FIELD)
    {
        sources
            .iter()
            .flat_map(|source| text_pairs(source.get(REQUEST_METADATA_CLIENT_FIELD)))
            .collect()
    } else {
        Vec::new()
    };
    let mut seen: Vec<&str> = Vec::new();
    let client: Vec<(String, String)> = spend_logs_pairs
        .into_iter()
        .filter(|(key, value)| !is_reserved(key) && is_forwardable(key, value))
        .filter(|(key, _)| {
            let first = !seen.contains(key);
            seen.push(key);
            first
        })
        .take(REQUEST_METADATA_MAX_PAIRS - identity.len())
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect();
    identity
        .into_iter()
        .chain(client)
        .fold(Vec::new(), |mut resolved, (key, value)| {
            match resolved.iter_mut().find(|(existing, _)| *existing == key) {
                Some(slot) => slot.1 = value,
                None => resolved.push((key, value)),
            }
            resolved
        })
}

fn header_value(pairs: &[(String, String)]) -> String {
    let object: Map<String, Value> = pairs
        .iter()
        .map(|(key, value)| (key.clone(), Value::String(value.clone())))
        .collect();
    python_json::dumps(&python_json::from_json(Value::Object(object)))
        .map(|text| compact_python_json(&text))
        .unwrap_or_default()
}

fn compact_python_json(text: &str) -> String {
    let mut compact = String::with_capacity(text.len());
    let mut in_string = false;
    let mut escaped = false;
    let mut characters = text.chars().peekable();
    while let Some(character) = characters.next() {
        if in_string {
            compact.push(character);
            match (escaped, character) {
                (true, _) => escaped = false,
                (false, '\\') => escaped = true,
                (false, '"') => in_string = false,
                _ => {}
            }
            continue;
        }
        match character {
            '"' => {
                in_string = true;
                compact.push(character);
            }
            ',' | ':' if characters.peek() == Some(&' ') => {
                compact.push(character);
                characters.next();
            }
            _ => compact.push(character),
        }
    }
    compact
}

fn longest_fitting_prefix(pairs: &[(String, String)]) -> Option<String> {
    (1..=pairs.len())
        .rev()
        .map(|count| header_value(&pairs[..count]))
        .find(|value| value.chars().count() <= REQUEST_METADATA_HEADER_MAX_CHARS)
}

pub(super) fn request_metadata_header(
    extra: &std::collections::BTreeMap<String, Value>,
) -> Option<Option<(String, String)>> {
    let allowed = allowed_fields(extra.get(REQUEST_METADATA_FIELDS_PARAM));
    if allowed.is_empty() {
        return None;
    }
    let sources: Vec<&Map<String, Value>> = METADATA_PARAMS
        .iter()
        .filter_map(|name| extra.get(*name).and_then(Value::as_object))
        .collect();
    let resolved = resolved_pairs(&sources, &allowed);
    Some(
        longest_fitting_prefix(&resolved).map(|value| (REQUEST_METADATA_HEADER.to_string(), value)),
    )
}

pub(super) fn merged_invoke_headers(
    headers: Headers,
    caller_owned: Vec<(String, String)>,
    proxy_owned: Option<Option<(String, String)>>,
) -> Headers {
    let owned_names: Vec<String> = proxy_owned
        .as_ref()
        .map(|_| vec![REQUEST_METADATA_HEADER.to_lowercase()])
        .unwrap_or_default();
    if caller_owned.is_empty() && owned_names.is_empty() {
        return headers;
    }
    let existing: Vec<String> = headers
        .iter()
        .map(|(name, _)| name.to_lowercase())
        .collect();
    let kept = headers
        .into_iter()
        .filter(|(name, _)| !owned_names.contains(&name.to_lowercase()));
    let added = caller_owned
        .into_iter()
        .filter(|(name, _)| !existing.contains(&name.to_lowercase()));
    kept.chain(added).chain(proxy_owned.flatten()).fold(
        Vec::new(),
        |mut merged: Headers, (name, value)| {
            match merged.iter_mut().find(|(existing, _)| *existing == name) {
                Some(slot) => slot.1 = value,
                None => merged.push((name, value)),
            }
            merged
        },
    )
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn extra(value: Value) -> std::collections::BTreeMap<String, Value> {
        value
            .as_object()
            .unwrap()
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect()
    }

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
            guardrail_headers(
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
            rejection(guardrail_headers(&json!({"guardrailVersion": "2"})).unwrap_err()),
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
        let (status, body) = rejection(guardrail_headers(&raw).unwrap_err());
        assert_eq!(status, 400);
        assert!(
            body.starts_with(&format!(
                "Invalid guardrailConfig={}. Expected format: ",
                python_repr(&raw)
            )),
            "{body}"
        );
    }

    #[test]
    fn a_callers_guardrail_header_wins() {
        let merged = merged_invoke_headers(
            vec![("x-amzn-bedrock-guardrailidentifier".into(), "caller".into())],
            vec![
                (GUARDRAIL_IDENTIFIER_HEADER.into(), "deployment".into()),
                (GUARDRAIL_VERSION_HEADER.into(), "1".into()),
            ],
            None,
        );
        assert_eq!(
            merged,
            vec![
                (
                    "x-amzn-bedrock-guardrailidentifier".to_string(),
                    "caller".to_string()
                ),
                (GUARDRAIL_VERSION_HEADER.to_string(), "1".to_string()),
            ]
        );
    }

    #[rstest]
    #[case::disabled_by_default(json!({}), None)]
    #[case::identity_then_spend_logs(
        json!({
            "bedrock_request_metadata_fields": ["user_api_key_team_id", "spend_logs_metadata", "user_api_key_team_id"],
            "metadata": {"user_api_key_team_id": "team-1", "spend_logs_metadata": {"project": "alpha", "user_api_key_alias": "forged", "bad key!": "x"}},
        }),
        Some(Some(r#"{"user_api_key_team_id":"team-1","project":"alpha"}"#)),
    )]
    #[case::litellm_metadata_is_read_too(
        json!({"bedrock_request_metadata_fields": ["user_api_key_user_id"], "litellm_metadata": {"user_api_key_user_id": "u 1"}}),
        Some(Some(r#"{"user_api_key_user_id":"u 1"}"#)),
    )]
    #[case::owned_but_empty(
        json!({"bedrock_request_metadata_fields": ["user_api_key_team_id"], "metadata": {}}),
        Some(None),
    )]
    fn request_metadata_header_follows_the_allow_list(
        #[case] params: Value,
        #[case] expected: Option<Option<&str>>,
    ) {
        assert_eq!(
            request_metadata_header(&extra(params)),
            expected.map(|header| header
                .map(|value| (REQUEST_METADATA_HEADER.to_string(), value.to_string())))
        );
    }

    #[test]
    fn the_proxy_owns_the_request_metadata_header() {
        let merged = merged_invoke_headers(
            vec![
                ("x-amzn-bedrock-request-metadata".into(), "forged".into()),
                ("x-keep".into(), "1".into()),
            ],
            Vec::new(),
            Some(None),
        );
        assert_eq!(merged, vec![("x-keep".to_string(), "1".to_string())]);
    }

    #[test]
    fn the_header_keeps_the_longest_prefix_that_fits() {
        let long = "v".repeat(256);
        let spend_logs: Map<String, Value> = (0..40)
            .map(|index| (format!("key{index:02}"), Value::String(long.clone())))
            .collect();
        let params = extra(json!({
            "bedrock_request_metadata_fields": ["spend_logs_metadata"],
            "metadata": {"spend_logs_metadata": spend_logs},
        }));
        let Some(Some((_, value))) = request_metadata_header(&params) else {
            panic!("expected a header");
        };
        let pairs: Map<String, Value> = serde_json::from_str(&value).unwrap();
        assert_eq!(pairs.len(), 16);
        assert!(value.len() <= REQUEST_METADATA_HEADER_MAX_CHARS);
    }
}
