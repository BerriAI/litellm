//! The credentials a `vertex_credentials` value names, read as Python's
//! `load_vertex_credentials_source` reads them and parsed once into typed credentials.

use std::path::Path;

use serde_json::{Map, Value};

use crate::{
    config::is_base64_encoded_json,
    google::{
        GoogleLoginFailure, PythonException, UnusableFile,
        credentials::{
            AuthorizedUserInfo, BoundedRead, CredentialsFile, CredentialsType, ExternalAccountInfo,
            MAX_CREDENTIALS_FILE_CHARACTERS, ServiceAccountInfo, Unparsed, read_bounded,
        },
    },
    identity::SdkJson,
};

const INLINE_JSON_LEADING_NOISE: [char; 7] = [' ', '\t', '\r', '\n', '\u{feff}', '"', '\''];
const NOT_A_JSON_OBJECT: &str = "is JSON but not a JSON object, it may be quoted or encoded twice";
const PRIVATE_KEY_HINT: &str = "Check for unescaped newlines in private_key.";
const LEADING_NOISE_HINT: &str = "Remove the quotes or byte-order mark before the opening `{`.";

/// The credentials `_credentials_from_json` builds for a `vertex_credentials` value: an
/// authorized user or an external account by type, and a service account for any other type.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ExplicitCredentials {
    AuthorizedUser(AuthorizedUserInfo),
    ExternalAccount(Box<ExternalAccountInfo>),
    ServiceAccount(ServiceAccountInfo),
}

impl ExplicitCredentials {
    fn parse(info: Map<String, Value>) -> Result<Self, serde_json::Error> {
        match CredentialsFile::parse(info.clone()) {
            Ok(CredentialsFile::AuthorizedUser(user)) => Ok(Self::AuthorizedUser(user)),
            Ok(CredentialsFile::ExternalAccount(account)) => Ok(Self::ExternalAccount(account)),
            Err(Unparsed::Field(error))
                if matches!(
                    CredentialsType::declared_by(&info),
                    Some(CredentialsType::AuthorizedUser | CredentialsType::ExternalAccount)
                ) =>
            {
                Err(error)
            }
            _ => serde_json::from_value(Value::Object(info.clone())).map(|account| {
                Self::ServiceAccount(ServiceAccountInfo {
                    sdk: SdkJson(info),
                    ..account
                })
            }),
        }
    }

    /// The project `load_auth` reads off the credentials: an authorized user's quota project, a
    /// service account's `project_id`, nothing for an external account.
    pub(crate) fn project(&self) -> Option<String> {
        match self {
            Self::AuthorizedUser(user) => user.quota_project_id.clone(),
            Self::ExternalAccount(_) => None,
            Self::ServiceAccount(account) => account.project_id.clone(),
        }
    }
}

/// The credentials a `vertex_credentials` value names, and the file it came from.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CredentialsJson {
    pub(crate) credentials: ExplicitCredentials,
    pub(crate) path: Option<String>,
}

/// The outcome of reading text as a credentials JSON object.
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
    GoogleLoginFailure::credentials_file(UnusableFile::Invalid {
        path: path.into(),
        problem,
    })
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
    let unreadable = |error: std::io::Error| {
        GoogleLoginFailure::credentials_file(UnusableFile::Unreadable {
            path: path.into(),
            reason: error.to_string(),
        })
    };
    let too_large = || {
        invalid_file(
            path,
            format!(
                "is over {MAX_CREDENTIALS_FILE_CHARACTERS} characters, larger than any credentials file"
            ),
        )
    };
    // At most four UTF-8 bytes a character, so one byte past that is over the limit whatever it
    // decodes to.
    let bytes = read_bounded(path).map_err(|failure| match failure {
        BoundedRead::Unreadable(error) => unreadable(error),
        BoundedRead::TooLarge => too_large(),
    })?;
    let contents =
        String::from_utf8(bytes).map_err(|_| invalid_file(path, "is not UTF-8 text".into()))?;
    if contents.chars().count() > MAX_CREDENTIALS_FILE_CHARACTERS {
        return Err(too_large());
    }
    match parse_json_object(&contents) {
        Parsed::Object(info) => Ok(CredentialsJson {
            credentials: ExplicitCredentials::parse(info).map_err(|error| {
                invalid_file(path, format!("holds a field of the wrong type: {error}"))
            })?,
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
        Parsed::Object(info) => Ok(CredentialsJson {
            credentials: ExplicitCredentials::parse(info)
                .map_err(|error| unusable(&format!("holds a field of the wrong type: {error}")))?,
            path: None,
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
        UnusableFile::Unreadable {
            path: "/nonexistent/vertex.json".into(),
            reason: "No such file or directory (os error 2)".into(),
        }
    )]
    #[case::directory(
        "/",
        UnusableFile::Unreadable {
            path: "/".into(),
            reason: "Is a directory (os error 21)".into(),
        }
    )]
    #[case::null_byte(
        "/tmp/a\0b",
        UnusableFile::Unreadable {
            path: "/tmp/a\0b".into(),
            reason: "file name contained an unexpected NUL byte".into(),
        }
    )]
    fn an_unreadable_file_names_its_path_and_the_reason(
        #[case] path: &str,
        #[case] expected: UnusableFile,
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
            GoogleLoginFailure::credentials_file(UnusableFile::Invalid {
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
            Some(UnusableFile::Invalid { problem, .. }) if problem == "is over 1048576 characters, larger than any credentials file"
        ));
    }

    #[rstest]
    #[case::service_account(json!({"type": "service_account", "private_key": "credential-value"}))]
    #[case::authorized_user(json!({
        "type": "authorized_user",
        "refresh_token": "credential-value",
        "client_secret": "credential-value",
        "rapt_token": "credential-value",
    }))]
    #[case::external_account(json!({
        "type": "external_account",
        "client_secret": "credential-value",
        "aws_secret_access_key": "credential-value",
        "aws_session_token": "credential-value",
        "credential_source": {"url": "https://example.com"},
    }))]
    fn credentials_debug_never_shows_a_secret(#[case] info: Value) {
        let json = load_vertex_credentials_source(&info.to_string()).unwrap();
        let debug = format!("{json:?}");
        assert!(!debug.contains("credential-value"), "{debug}");
        assert!(debug.contains("[REDACTED]"), "{debug}");
    }

    #[cfg(unix)]
    #[rstest]
    fn an_endless_file_is_refused_by_its_size_without_reading_it_all() {
        let failure = load_vertex_credentials_source("/dev/zero").unwrap_err();
        assert!(matches!(
            failure.file,
            Some(UnusableFile::Invalid { problem, .. }) if problem == "is over 1048576 characters, larger than any credentials file"
        ));
    }

    #[rstest]
    #[case::file(true)]
    #[case::inline(false)]
    fn inline_json_and_a_file_are_read_as_the_credentials_their_type_names(#[case] file: bool) {
        let info = json!({"type": "service_account", "project_id": "p", "private_key_id": "k"});
        let key = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(key.path(), info.to_string()).unwrap();
        let path = key.path().to_str().unwrap();
        let read = load_vertex_credentials_source(match file {
            true => path,
            false => {
                " {\"type\":\"service_account\",\"project_id\":\"p\",\"private_key_id\":\"k\"}"
            }
        })
        .unwrap();
        assert_eq!(
            read,
            CredentialsJson {
                credentials: ExplicitCredentials::ServiceAccount(ServiceAccountInfo {
                    project_id: Some("p".into()),
                    private_key_id: Some("k".into()),
                    sdk: SdkJson(info.as_object().unwrap().clone()),
                    ..ServiceAccountInfo::default()
                }),
                path: file.then(|| path.into()),
            }
        );
    }

    #[rstest]
    #[case::untyped(json!({"project_id": "p"}))]
    #[case::unknown_type(json!({"type": "gdch_service_account", "project_id": "p"}))]
    #[case::impersonated(json!({"type": "impersonated_service_account", "project_id": "p"}))]
    fn any_other_type_is_read_as_a_service_account(#[case] info: Value) {
        let credentials = ExplicitCredentials::parse(info.as_object().unwrap().clone()).unwrap();
        assert_eq!(credentials.project().as_deref(), Some("p"));
        assert!(matches!(
            credentials,
            ExplicitCredentials::ServiceAccount(_)
        ));
    }

    #[rstest]
    #[case::numeric_project(json!({"type": "service_account", "project_id": 7}))]
    #[case::numeric_quota_project(json!({"type": "authorized_user", "quota_project_id": 7}))]
    #[case::listed_audience(json!({"type": "external_account", "audience": ["a"]}))]
    fn a_field_of_the_wrong_type_is_refused_by_name(#[case] info: Value) {
        let failure = load_vertex_credentials_source(&info.to_string()).unwrap_err();
        assert_eq!(failure.exception, PythonException::Value);
        assert!(
            failure.message.starts_with(
                "The `vertex_credentials` value holds a field of the wrong type: invalid type"
            ),
            "{}",
            failure.message
        );
    }

    #[rstest]
    fn a_file_with_a_field_of_the_wrong_type_names_its_path() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            file.path(),
            r#"{"type":"service_account","private_key_id":1}"#,
        )
        .unwrap();
        let failure = load_vertex_credentials_source(file.path().to_str().unwrap()).unwrap_err();
        assert!(matches!(
            failure.file,
            Some(UnusableFile::Invalid { problem, .. }) if problem.starts_with("holds a field of the wrong type: invalid type: integer `1`")
        ));
    }
}
