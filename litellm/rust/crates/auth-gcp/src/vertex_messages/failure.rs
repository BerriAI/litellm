//! A Vertex AI Messages login failure: the exception class Python's `VertexBase.load_auth`
//! raises for it, its text, and the credentials file it is about, whose path only the log names.

use litellm_auth_types::Error;

pub(crate) const FILE_FAILURE_MESSAGE: &str =
    "Unable to load the vertex credentials file. LiteLLM's log names the path and the reason.";

/// The exception class Python's Vertex AI login raises, by the name a host raises it under.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
pub enum PythonException {
    #[strum(serialize = "ValueError")]
    Value,
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
pub enum UnusableFile {
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
    pub file: Option<UnusableFile>,
    /// Google's token endpoint refused the grant for good, so loading the credentials again
    /// (a re-run `gcloud auth application-default login` or a rotated key) may fix it.
    pub grant_rejected: bool,
}

impl GoogleLoginFailure {
    pub fn new(exception: PythonException, message: impl Into<String>) -> Self {
        Self {
            exception,
            message: message.into(),
            file: None,
            grant_rejected: false,
        }
    }

    /// The same failure, marked as a grant Google refused for good.
    pub(crate) fn rejecting_the_grant(self) -> Self {
        Self {
            grant_rejected: true,
            ..self
        }
    }

    /// A credentials file that could not be used: its path and the reason go to LiteLLM's log,
    /// and the failure names neither.
    pub(crate) fn credentials_file(file: UnusableFile) -> Self {
        match &file {
            UnusableFile::Unreadable { path, reason } => tracing::error!(
                "Vertex: cannot read the credentials file at {}: {reason}",
                path_for_log(path)
            ),
            UnusableFile::Invalid { path, problem } => tracing::error!(
                "Vertex: the credentials file at {} {problem}",
                path_for_log(path)
            ),
        }
        Self {
            file: Some(file),
            ..Self::new(PythonException::Value, FILE_FAILURE_MESSAGE)
        }
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
    format!("'{}'", REDACTOR.redact(path))
}

impl From<GoogleLoginFailure> for Error {
    fn from(failure: GoogleLoginFailure) -> Self {
        Error::provider(failure)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

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
        UnusableFile::Unreadable { path: "/etc/vertex/key.json".into(), reason: "No such file or directory (os error 2)".into() },
        "Vertex: cannot read the credentials file at '/etc/vertex/key.json': No such file or directory (os error 2)"
    )]
    #[case::invalid(
        UnusableFile::Invalid { path: "/etc/vertex/key.json".into(), problem: "is not UTF-8 text".into() },
        "Vertex: the credentials file at '/etc/vertex/key.json' is not UTF-8 text"
    )]
    #[case::long_path(
        UnusableFile::Invalid { path: "a".repeat(261), problem: "is empty".into() },
        "Vertex: the credentials file at <a 261-character value> is empty"
    )]
    fn a_credentials_file_failure_logs_the_path_and_raises_only_the_file_failure(
        #[case] file: UnusableFile,
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
}
