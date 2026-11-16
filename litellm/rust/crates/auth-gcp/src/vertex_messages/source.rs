use std::path::Path;

use serde_json::{Map, Value};

use super::failure::{CredentialsFile, GoogleLoginFailure, PythonException};
use crate::config::is_base64_encoded_json;

const INLINE_JSON_LEADING_NOISE: [char; 7] = [' ', '\t', '\r', '\n', '\u{feff}', '"', '\''];
const MAX_CREDENTIALS_FILE_CHARACTERS: usize = 1 << 20;
const NOT_A_JSON_OBJECT: &str = "is JSON but not a JSON object, it may be quoted or encoded twice";
const PRIVATE_KEY_HINT: &str = "Check for unescaped newlines in private_key.";
const LEADING_NOISE_HINT: &str = "Remove the quotes or byte-order mark before the opening `{`.";

/// The credentials JSON a `vertex_credentials` value names, and the file it came from, as
/// Python's `load_vertex_credentials_source` reads it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CredentialsJson {
    pub(crate) info: Map<String, Value>,
    pub(crate) path: Option<String>,
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

fn invalid_file(path: &str, problem: String) -> GoogleLoginFailure {
    GoogleLoginFailure::credentials_file(CredentialsFile::Invalid {
        path: path.into(),
        problem,
    })
}

/// Python's `f"{e.strerror} ({type(e).__name__})"` for the `OSError` `open()` raises.
fn unreadable_reason(path: &str, error: &std::io::Error) -> String {
    if path.contains('\0') {
        return "embedded null byte (ValueError)".into();
    }
    let class = match error.kind() {
        std::io::ErrorKind::NotFound => "FileNotFoundError",
        std::io::ErrorKind::PermissionDenied => "PermissionError",
        std::io::ErrorKind::IsADirectory => "IsADirectoryError",
        std::io::ErrorKind::NotADirectory => "NotADirectoryError",
        _ => "OSError",
    };
    let text = error.to_string();
    let strerror = text
        .rsplit_once(" (os error ")
        .map_or(text.as_str(), |(strerror, _)| strerror);
    format!("{strerror} ({class})")
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
    let bytes = std::fs::read(path).map_err(|error| {
        GoogleLoginFailure::credentials_file(CredentialsFile::Unreadable {
            path: path.into(),
            reason: unreadable_reason(path, &error),
        })
    })?;
    let contents =
        String::from_utf8(bytes).map_err(|_| invalid_file(path, "is not UTF-8 text".into()))?;
    if contents.chars().count() > MAX_CREDENTIALS_FILE_CHARACTERS {
        return Err(invalid_file(
            path,
            format!(
                "is over {MAX_CREDENTIALS_FILE_CHARACTERS} characters, larger than any credentials file"
            ),
        ));
    }
    match parse_json_object(&contents) {
        Parsed::Object(info) => Ok(CredentialsJson {
            info,
            path: Some(path.into()),
        }),
        Parsed::Invalid(detail) => Err(invalid_file(
            path,
            format!("is not valid JSON: {detail}. {PRIVATE_KEY_HINT}"),
        )),
        Parsed::NotAnObject => Err(invalid_file(path, NOT_A_JSON_OBJECT.into())),
    }
}

fn load_inline_credentials(credentials: &str) -> Result<CredentialsJson, GoogleLoginFailure> {
    match parse_json_object(credentials) {
        Parsed::Object(info) => Ok(CredentialsJson { info, path: None }),
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

    #[rstest]
    #[case::empty(
        "",
        "The `vertex_credentials` value is empty. Set it to a readable file path, or to the credentials JSON itself."
    )]
    #[case::blank(
        "   ",
        "The `vertex_credentials` value is empty. Set it to a readable file path, or to the credentials JSON itself."
    )]
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
    fn unusable_values_raise_python_s_value_error(
        #[case] credentials: &str,
        #[case] expected: &str,
    ) {
        let failure = load_vertex_credentials_source(credentials).unwrap_err();
        assert_eq!(failure.exception, PythonException::Value);
        assert_eq!(failure.message, expected);
    }

    #[rstest]
    #[case::missing(
        "/nonexistent/vertex.json",
        CredentialsFile::Unreadable {
            path: "/nonexistent/vertex.json".into(),
            reason: "No such file or directory (FileNotFoundError)".into(),
        }
    )]
    #[case::directory(
        "/",
        CredentialsFile::Unreadable {
            path: "/".into(),
            reason: "Is a directory (IsADirectoryError)".into(),
        }
    )]
    #[case::null_byte(
        "/tmp/a\0b",
        CredentialsFile::Unreadable {
            path: "/tmp/a\0b".into(),
            reason: "embedded null byte (ValueError)".into(),
        }
    )]
    fn an_unreadable_file_names_its_path_and_python_s_os_error(
        #[case] path: &str,
        #[case] expected: CredentialsFile,
    ) {
        assert_eq!(
            load_vertex_credentials_source(path).unwrap_err(),
            GoogleLoginFailure::credentials_file(expected)
        );
    }

    #[rstest]
    #[case::not_utf8(b"\xff{".as_slice(), "is not UTF-8 text")]
    #[case::not_json(b"{oops".as_slice(), "is not valid JSON: Invalid JSON: key must be a string at line 1 column 2. Check for unescaped newlines in private_key.")]
    #[case::not_an_object(b"[1]".as_slice(), NOT_A_JSON_OBJECT)]
    fn an_unusable_file_names_its_path_and_problem(#[case] contents: &[u8], #[case] problem: &str) {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), contents).unwrap();
        let path = file.path().to_str().unwrap();
        assert_eq!(
            load_vertex_credentials_source(path).unwrap_err(),
            GoogleLoginFailure::credentials_file(CredentialsFile::Invalid {
                path: path.into(),
                problem: problem.into(),
            })
        );
    }

    #[rstest]
    fn a_file_over_the_size_limit_is_refused_by_its_size() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), " ".repeat(MAX_CREDENTIALS_FILE_CHARACTERS + 1)).unwrap();
        let failure = load_vertex_credentials_source(file.path().to_str().unwrap()).unwrap_err();
        assert!(matches!(
            failure.file,
            Some(CredentialsFile::Invalid { problem, .. }) if problem == "is over 1048576 characters, larger than any credentials file"
        ));
    }

    #[rstest]
    fn inline_json_and_a_file_are_read_as_objects() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), r#"{"type":"service_account"}"#).unwrap();
        assert_eq!(
            load_vertex_credentials_source(file.path().to_str().unwrap()).unwrap(),
            CredentialsJson {
                info: json!({"type": "service_account"})
                    .as_object()
                    .unwrap()
                    .clone(),
                path: Some(file.path().to_str().unwrap().into())
            }
        );
        assert_eq!(
            load_vertex_credentials_source(r#" {"type":"authorized_user"}"#).unwrap(),
            CredentialsJson {
                info: json!({"type": "authorized_user"})
                    .as_object()
                    .unwrap()
                    .clone(),
                path: None
            }
        );
    }
}
