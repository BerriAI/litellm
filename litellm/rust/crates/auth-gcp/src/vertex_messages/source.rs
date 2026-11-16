use std::path::Path;

use serde_json::{Map, Value};

use super::failure::{GoogleLoginFailure, PythonException};
use crate::config::is_base64_encoded_json;

const INLINE_JSON_LEADING_NOISE: [char; 7] = [' ', '\t', '\r', '\n', '\u{feff}', '"', '\''];
const MAX_CREDENTIALS_FILE_CHARACTERS: usize = 1 << 20;
const NOT_A_JSON_OBJECT: &str = "is JSON but not a JSON object, it may be quoted or encoded twice";
const PRIVATE_KEY_HINT: &str = "Check for unescaped newlines in private_key.";
const LEADING_NOISE_HINT: &str = "Remove the quotes or byte-order mark before the opening `{`.";
pub(crate) const FILE_FAILURE_MESSAGE: &str =
    "Unable to load the vertex credentials file. LiteLLM's log names the path and the reason.";

/// The credentials JSON a `vertex_credentials` value names, and whether it came from a file,
/// as Python's `load_vertex_credentials_source` reads it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CredentialsJson {
    pub(crate) info: Map<String, Value>,
    pub(crate) from_file: bool,
}

enum Parsed {
    Object(Map<String, Value>),
    Invalid(String),
    NotAnObject,
}

fn parse_json_object(raw: &str) -> Parsed {
    match serde_json::from_str::<Value>(raw) {
        Ok(Value::Object(info)) => Parsed::Object(info),
        Ok(_) => Parsed::NotAnObject,
        Err(error) => Parsed::Invalid(format!("Invalid JSON: {error}")),
    }
}

fn file_failure() -> GoogleLoginFailure {
    GoogleLoginFailure::new(PythonException::Value, FILE_FAILURE_MESSAGE)
}

fn unusable(problem: &str) -> GoogleLoginFailure {
    GoogleLoginFailure::new(
        PythonException::Value,
        format!(
            "The `vertex_credentials` value {problem}. Set it to a readable file path, or to the credentials JSON itself."
        ),
    )
}

fn inline_not_json(detail: &str, hint: &str) -> GoogleLoginFailure {
    GoogleLoginFailure::new(
        PythonException::Value,
        format!("The inline `vertex_credentials` value is not valid JSON: {detail}. {hint}"),
    )
}

fn read_json_file(path: &str) -> Result<CredentialsJson, GoogleLoginFailure> {
    let contents = std::fs::read_to_string(path).map_err(|_| file_failure())?;
    if contents.chars().count() > MAX_CREDENTIALS_FILE_CHARACTERS {
        return Err(file_failure());
    }
    match parse_json_object(&contents) {
        Parsed::Object(info) => Ok(CredentialsJson {
            info,
            from_file: true,
        }),
        Parsed::Invalid(_) | Parsed::NotAnObject => Err(file_failure()),
    }
}

fn load_inline_credentials(credentials: &str) -> Result<CredentialsJson, GoogleLoginFailure> {
    match parse_json_object(credentials) {
        Parsed::Object(info) => Ok(CredentialsJson {
            info,
            from_file: false,
        }),
        _ if Path::new(credentials).is_file() => read_json_file(credentials),
        Parsed::NotAnObject => Err(unusable(NOT_A_JSON_OBJECT)),
        Parsed::Invalid(detail) if credentials.trim_start().starts_with('{') => {
            Err(inline_not_json(&detail, PRIVATE_KEY_HINT))
        }
        Parsed::Invalid(detail) => Err(inline_not_json(&detail, LEADING_NOISE_HINT)),
    }
}

/// Python's `load_vertex_credentials_source`: inline JSON, else base64-encoded JSON refused by
/// name, else a file to read.
pub(crate) fn load_vertex_credentials_source(
    credentials: &str,
) -> Result<CredentialsJson, GoogleLoginFailure> {
    if credentials.trim().is_empty() {
        return Err(unusable("is empty"));
    }
    if credentials
        .trim_start_matches(INLINE_JSON_LEADING_NOISE)
        .starts_with('{')
    {
        return load_inline_credentials(credentials);
    }
    if is_base64_encoded_json(credentials) {
        return Err(unusable("is base64-encoded JSON, not the JSON itself"));
    }
    read_json_file(credentials)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn message(credentials: &str) -> String {
        load_vertex_credentials_source(credentials).unwrap_err().message
    }

    #[rstest]
    #[case::empty("", "The `vertex_credentials` value is empty. Set it to a readable file path, or to the credentials JSON itself.")]
    #[case::blank("   ", "The `vertex_credentials` value is empty. Set it to a readable file path, or to the credentials JSON itself.")]
    #[case::base64(
        "eyJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCJ9",
        "The `vertex_credentials` value is base64-encoded JSON, not the JSON itself. Set it to a readable file path, or to the credentials JSON itself."
    )]
    #[case::invalid_inline(
        "{not json",
        "The inline `vertex_credentials` value is not valid JSON: Invalid JSON: key must be a string at line 1 column 2. Check for unescaped newlines in private_key."
    )]
    #[case::quoted_inline(
        "'{\"type\": 1",
        "The inline `vertex_credentials` value is not valid JSON: Invalid JSON: expected value at line 1 column 1. Remove the quotes or byte-order mark before the opening `{`."
    )]
    #[case::double_encoded(
        "\"{\\\"type\\\": 1}\"",
        "The `vertex_credentials` value is JSON but not a JSON object, it may be quoted or encoded twice. Set it to a readable file path, or to the credentials JSON itself."
    )]
    #[case::missing_file("/nonexistent/vertex.json", FILE_FAILURE_MESSAGE)]
    fn unusable_values_raise_python_s_value_error(#[case] credentials: &str, #[case] expected: &str) {
        let failure = load_vertex_credentials_source(credentials).unwrap_err();
        assert_eq!(failure.exception, PythonException::Value);
        assert_eq!(failure.message, expected);
    }

    #[rstest]
    #[case::not_json("{oops")]
    #[case::not_an_object("[1]")]
    fn an_unusable_file_raises_the_file_failure(#[case] contents: &str) {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), contents).unwrap();
        assert_eq!(message(file.path().to_str().unwrap()), FILE_FAILURE_MESSAGE);
    }

    #[rstest]
    fn inline_json_and_a_file_are_read_as_objects() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), r#"{"type":"service_account"}"#).unwrap();
        assert_eq!(
            load_vertex_credentials_source(file.path().to_str().unwrap()).unwrap(),
            CredentialsJson {
                info: json!({"type": "service_account"}).as_object().unwrap().clone(),
                from_file: true
            }
        );
        assert_eq!(
            load_vertex_credentials_source(r#" {"type":"authorized_user"}"#).unwrap(),
            CredentialsJson {
                info: json!({"type": "authorized_user"}).as_object().unwrap().clone(),
                from_file: false
            }
        );
    }
}
