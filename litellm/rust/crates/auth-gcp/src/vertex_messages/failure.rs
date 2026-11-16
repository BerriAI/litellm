use litellm_auth_types::Error;
use serde_json::Value;

/// Python's `GOOGLE_IMPORT_ERROR_MESSAGE`, which its login raises when google-auth is not installed.
pub const GOOGLE_SDK_MISSING: &str = "Google Cloud SDK not found. Install it with: pip install 'litellm[google]' or pip install google-cloud-aiplatform";

pub(crate) const FILE_FAILURE_MESSAGE: &str =
    "Unable to load the vertex credentials file. LiteLLM's log names the path and the reason.";

/// The exception class Python's Vertex AI login raises, by the name a host raises it under.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
pub enum PythonException {
    #[strum(serialize = "ImportError")]
    Import,
    #[strum(serialize = "ValueError")]
    Value,
    #[strum(serialize = "TypeError")]
    Type,
    #[strum(serialize = "AttributeError")]
    Attribute,
    #[strum(serialize = "IndexError")]
    Index,
    #[strum(serialize = "OSError")]
    OS,
    #[strum(serialize = "google.auth.exceptions.DefaultCredentialsError")]
    DefaultCredentials,
    #[strum(serialize = "google.auth.exceptions.MalformedError")]
    Malformed,
    #[strum(serialize = "google.auth.exceptions.InvalidValue")]
    InvalidValue,
    #[strum(serialize = "google.auth.exceptions.InvalidResource")]
    InvalidResource,
    #[strum(serialize = "google.auth.exceptions.RefreshError")]
    Refresh,
    #[strum(serialize = "google.auth.exceptions.TransportError")]
    Transport,
    #[strum(serialize = "google.auth.exceptions.OAuthError")]
    OAuth,
}

/// A `vertex_credentials` file Python's `raise_vertex_credentials_failure` logs by path before it
/// raises its one file failure text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CredentialsFile {
    Unreadable { path: String, reason: String },
    Invalid { path: String, problem: String },
}

/// A Vertex AI Messages login failure, as the exception Python's `VertexBase.load_auth` raises
/// for it: its class and its text.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct GoogleLoginFailure {
    pub exception: PythonException,
    pub message: String,
    pub file: Option<CredentialsFile>,
}

impl GoogleLoginFailure {
    pub fn new(exception: PythonException, message: impl Into<String>) -> Self {
        Self {
            exception,
            message: message.into(),
            file: None,
        }
    }

    /// A credentials file that could not be used: its path and the reason go to LiteLLM's log,
    /// and the failure names neither.
    pub(crate) fn credentials_file(file: CredentialsFile) -> Self {
        match &file {
            CredentialsFile::Unreadable { path, reason } => tracing::error!(
                "Vertex: cannot read the credentials file at {}: {reason}",
                path_for_log(path)
            ),
            CredentialsFile::Invalid { path, problem } => tracing::error!(
                "Vertex: the credentials file at {} {problem}",
                path_for_log(path)
            ),
        }
        Self {
            file: Some(file),
            ..Self::new(PythonException::Value, FILE_FAILURE_MESSAGE)
        }
    }

    /// Python's `str()` of an exception raised with a message and its cause as two arguments.
    pub(crate) fn caused_by(exception: PythonException, message: &str, cause: &Self) -> Self {
        Self::new(
            exception,
            format!(
                "({}, {}({}))",
                python_repr(message),
                short_name(cause.exception),
                python_repr(&cause.message)
            ),
        )
    }
}

/// Paths longer than this are logged by their length only.
const MAX_LOGGED_PATH_LENGTH: usize = 260;
/// LiteLLM's default `MINIMUM_CUSTOM_KEY_LENGTH`, which its secret redaction uses.
const MINIMUM_CUSTOM_KEY_LENGTH: usize = 16;

/// A credentials file's path as LiteLLM logs it: quoted with secrets redacted, or only its
/// length when it is too long to be a path.
fn path_for_log(path: &str) -> String {
    static REDACTOR: std::sync::LazyLock<litellm_tracing::SecretRedactor> =
        std::sync::LazyLock::new(|| {
            litellm_tracing::SecretRedactor::new(MINIMUM_CUSTOM_KEY_LENGTH)
        });
    let length = path.chars().count();
    if length > MAX_LOGGED_PATH_LENGTH {
        return format!("<a {length}-character value>");
    }
    python_repr(&REDACTOR.redact(path))
}

impl From<GoogleLoginFailure> for Error {
    fn from(failure: GoogleLoginFailure) -> Self {
        Error::provider(failure)
    }
}

fn short_name(exception: PythonException) -> &'static str {
    let name: &'static str = exception.into();
    name.rsplit('.').next().unwrap_or(name)
}

/// Python's `repr()` of a string: single quotes unless the text holds one and no double quote.
pub(crate) fn python_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let escaped: String = text
        .chars()
        .flat_map(|character| match character {
            '\\' => vec!['\\', '\\'],
            '\n' => vec!['\\', 'n'],
            '\r' => vec!['\\', 'r'],
            '\t' => vec!['\\', 't'],
            character if character == quote => vec!['\\', character],
            character => vec![character],
        })
        .collect();
    format!("{quote}{escaped}{quote}")
}

/// Python's `str(type(value))` for a value parsed from JSON.
pub(crate) fn python_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "<class 'NoneType'>",
        Value::Bool(_) => "<class 'bool'>",
        Value::Number(number) if number.is_f64() => "<class 'float'>",
        Value::Number(_) => "<class 'int'>",
        Value::String(_) => "<class 'str'>",
        Value::Array(_) => "<class 'list'>",
        Value::Object(_) => "<class 'dict'>",
    }
}

/// Python's `type(value).__name__` for a value parsed from JSON, `NoneType` for a missing one.
pub(crate) fn python_type_name(value: Option<&Value>) -> &'static str {
    match value {
        None | Some(Value::Null) => "NoneType",
        Some(Value::Bool(_)) => "bool",
        Some(Value::Number(number)) if number.is_f64() => "float",
        Some(Value::Number(_)) => "int",
        Some(Value::String(_)) => "str",
        Some(Value::Array(_)) => "list",
        Some(Value::Object(_)) => "dict",
    }
}

/// Python's `str()` of a JSON value interpolated into a message.
pub(crate) fn python_str(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => "None".into(),
        Some(Value::String(text)) => text.clone(),
        Some(Value::Bool(flag)) => if *flag { "True" } else { "False" }.into(),
        Some(other) => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[derive(Clone, Default)]
    struct Errors(std::sync::Arc<std::sync::Mutex<Vec<String>>>);

    impl litellm_tracing::Sink for Errors {
        fn enabled(&self, metadata: &litellm_tracing::Metadata<'_>) -> bool {
            *metadata.level() == litellm_tracing::Level::ERROR
        }

        fn emit(&self, record: &litellm_tracing::Record) {
            self.0.lock().unwrap().push(record.message.clone());
        }
    }

    #[rstest]
    #[case::unreadable(
        CredentialsFile::Unreadable { path: "/etc/vertex/key.json".into(), reason: "No such file or directory (FileNotFoundError)".into() },
        "Vertex: cannot read the credentials file at '/etc/vertex/key.json': No such file or directory (FileNotFoundError)"
    )]
    #[case::invalid(
        CredentialsFile::Invalid { path: "/etc/vertex/key.json".into(), problem: "is not UTF-8 text".into() },
        "Vertex: the credentials file at '/etc/vertex/key.json' is not UTF-8 text"
    )]
    #[case::long_path(
        CredentialsFile::Invalid { path: "a".repeat(261), problem: "is empty".into() },
        "Vertex: the credentials file at <a 261-character value> is empty"
    )]
    fn a_credentials_file_failure_logs_the_path_and_raises_only_the_file_failure(
        #[case] file: CredentialsFile,
        #[case] logged: &str,
    ) {
        let errors = Errors::default();
        let failure = litellm_tracing::Logger::new(errors.clone())
            .scope(|| GoogleLoginFailure::credentials_file(file));
        assert_eq!(
            (failure.exception, failure.message.as_str()),
            (PythonException::Value, FILE_FAILURE_MESSAGE)
        );
        assert_eq!(*errors.0.lock().unwrap(), [logged.to_owned()]);
    }

    #[rstest]
    #[case::plain("File a.json", "'File a.json'")]
    #[case::single_quote("it's", "\"it's\"")]
    #[case::both_quotes("it's \"x\"", "'it\\'s \"x\"'")]
    #[case::newline("a\nb", "'a\\nb'")]
    fn strings_are_quoted_as_python_repr_quotes_them(#[case] text: &str, #[case] expected: &str) {
        assert_eq!(python_repr(text), expected);
    }

    #[rstest]
    fn a_wrapped_failure_reads_as_python_s_two_argument_exception() {
        let cause =
            GoogleLoginFailure::new(PythonException::Malformed, "missing fields token_uri.");
        let wrapped = GoogleLoginFailure::caused_by(
            PythonException::DefaultCredentials,
            "Failed to load service account credentials from /k.json",
            &cause,
        );
        assert_eq!(
            wrapped.message,
            "('Failed to load service account credentials from /k.json', MalformedError('missing fields token_uri.'))"
        );
    }

    #[rstest]
    #[case::int(json!(1), "<class 'int'>")]
    #[case::float(json!(1.5), "<class 'float'>")]
    #[case::boolean(json!(true), "<class 'bool'>")]
    #[case::list(json!(["p"]), "<class 'list'>")]
    #[case::object(json!({"p": 1}), "<class 'dict'>")]
    fn json_values_name_their_python_type(#[case] value: Value, #[case] expected: &str) {
        assert_eq!(python_type(&value), expected);
    }

    #[rstest]
    #[case::missing(None, "None")]
    #[case::null(Some(json!(null)), "None")]
    #[case::text(Some(json!("weird")), "weird")]
    #[case::number(Some(json!(3)), "3")]
    #[case::boolean(Some(json!(false)), "False")]
    fn json_values_print_as_python_prints_them(
        #[case] value: Option<Value>,
        #[case] expected: &str,
    ) {
        assert_eq!(python_str(value.as_ref()), expected);
    }
}
