//! A Google login failure: the google-auth exception class Python raises for it, its text, and the credentials file it is about, whose path only the log names.
//! Failures of Google's Rust SDK map onto the class google-auth raises for the same category.

use std::error::Error as StdError;

use google_cloud_auth::errors::{CredentialsError, SubjectTokenProviderError};
use litellm_auth_types::Error;

use super::environment::GoogleEnvironment;

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

/// A Google login failure, as the google-auth exception Python raises for it: its class and its
/// text.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct GoogleLoginFailure {
    pub exception: PythonException,
    pub message: String,
    pub file: Option<UnusableFile>,
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

impl SubjectTokenProviderError for GoogleLoginFailure {
    fn is_transient(&self) -> bool {
        self.exception == PythonException::Transport
    }
}

/// A service account key that could not sign its token, which google-auth refuses as a
/// `ValueError` when it loads the key.
#[derive(Debug, thiserror::Error)]
#[error("the service account key cannot sign a token")]
pub(crate) struct UnsignableKey(#[source] pub(crate) Error);

/// The start of the message Google's Rust SDK gives an STS token exchange that failed, which
/// google-auth's `sts.Client` raises as an `OAuthError` when STS refuses it.
const STS_EXCHANGE_FAILED: &str = "failed to exchange token";

// The SDK's own error text can quote a credential source's whole reply or file, so a token
// failure reaches the caller only as one of these fixed messages.
const TOKEN_ENDPOINT_UNREACHABLE: &str = "Vertex: the Google token endpoint could not be reached.";
const EXCHANGE_REFUSED: &str =
    "Vertex: Google's token exchange refused the external account credentials.";
const KEY_CANNOT_SIGN: &str = "Vertex: the service account private key cannot sign a token.";
const REFRESH_FAILED: &str = "Vertex: Google refused or could not complete the token refresh.";
/// A credentials document Google's Rust SDK cannot load, whose own error text may quote it.
pub(crate) const UNLOADABLE_CREDENTIALS: &str =
    "Vertex: Google's Rust SDK cannot load these credentials.";

const CA_BUNDLE_HINT: &str = "Google's Rust SDK verifies TLS against SSL_CERT_FILE, SSL_CERT_DIR or the system's certificates, never REQUESTS_CA_BUNDLE or CURL_CA_BUNDLE: set SSL_CERT_FILE to the same CA bundle.";

fn causes<'a>(
    error: &'a (dyn StdError + 'static),
) -> impl Iterator<Item = &'a (dyn StdError + 'static)> {
    std::iter::successors(Some(error), |&cause| cause.source())
}

/// A TLS failure to verify the server's certificate, which reqwest's connector wraps in
/// `io::Error`s that hide it from `source()`.
fn rejects_the_certificate(error: &(dyn StdError + 'static)) -> bool {
    causes(error)
        .flat_map(|cause| {
            std::iter::successors(Some(cause), |&wrapped| {
                wrapped
                    .downcast_ref::<std::io::Error>()
                    .and_then(std::io::Error::get_ref)
                    .map(|inner| inner as &(dyn StdError + 'static))
            })
        })
        .any(|cause| {
            matches!(
                cause.downcast_ref::<rustls::Error>(),
                Some(rustls::Error::InvalidCertificate(_))
            )
        })
}

/// Whether `requests` would have trusted a CA bundle the SDK does not read.
fn names_a_requests_only_bundle(env: &dyn GoogleEnvironment) -> bool {
    let set = |name: &str| env.os_env(name).is_some_and(|value| !value.is_empty());
    (set("REQUESTS_CA_BUNDLE") || set("CURL_CA_BUNDLE")) && !set("SSL_CERT_FILE")
}

impl GoogleLoginFailure {
    /// The exception google-auth raises when a token request fails: a `TransportError` when the
    /// endpoint could not be reached, an `OAuthError` when STS refused the exchange, a
    /// `ValueError` when a service account key cannot sign, else a `RefreshError`.
    pub(crate) fn token(error: &(dyn StdError + 'static), env: &dyn GoogleEnvironment) -> Self {
        if let Some(failure) = litellm_error::find::<Self>(error) {
            return failure.clone();
        }
        let http = litellm_error::find::<reqwest::Error>(error);
        if http.is_some_and(|http| http.status().is_none() && !http.is_decode()) {
            let hint = (rejects_the_certificate(error) && names_a_requests_only_bundle(env))
                .then_some(CA_BUNDLE_HINT);
            return Self::new(
                PythonException::Transport,
                [Some(TOKEN_ENDPOINT_UNREACHABLE), hint]
                    .into_iter()
                    .flatten()
                    .collect::<Vec<_>>()
                    .join(" "),
            );
        }
        let exchange_refused = http.is_some_and(|http| http.status().is_some())
            && causes(error).any(|cause| {
                cause
                    .downcast_ref::<CredentialsError>()
                    .is_some_and(|sdk| sdk.to_string().starts_with(STS_EXCHANGE_FAILED))
            });
        let exception = match (
            exchange_refused,
            litellm_error::find::<UnsignableKey>(error).is_some(),
        ) {
            (true, _) => (PythonException::OAuth, EXCHANGE_REFUSED),
            (false, true) => (PythonException::Value, KEY_CANNOT_SIGN),
            (false, false) => (PythonException::Refresh, REFRESH_FAILED),
        };
        Self::new(exception.0, exception.1)
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
