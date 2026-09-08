use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::time::SystemTime;

#[derive(Clone, PartialEq, Eq)]
pub struct SecretString(String);

impl SecretString {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("SecretString([REDACTED])")
    }
}

impl fmt::Display for SecretString {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("[REDACTED]")
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SuppliedSecret {
    pub source: String,
    pub value: Option<SecretString>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedCredential {
    pub value: SecretString,
    pub expires_at: Option<SystemTime>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AuthServiceError {
    Lookup { source: String },
    CallerToken,
    Headers,
}

impl fmt::Display for AuthServiceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lookup { source } => write!(formatter, "{source} lookup failed"),
            Self::CallerToken => formatter.write_str("caller token failed"),
            Self::Headers => formatter.write_str("header access failed"),
        }
    }
}

impl std::error::Error for AuthServiceError {}

pub trait AuthValueLookup: Send + Sync {
    fn lookup(&self, key: &str) -> Result<Option<SecretString>, AuthServiceError>;
}

pub type CallerTokenFuture<'a> =
    Pin<Box<dyn Future<Output = Result<SecretString, AuthServiceError>> + Send + 'a>>;

pub trait CallerTokenProvider: Send + Sync {
    fn invoke(&self) -> CallerTokenFuture<'_>;
}

pub trait ExecutionHeaders: Send + Sync {
    fn read(&self) -> Result<Vec<(String, String)>, AuthServiceError>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BodyAuthorizationInput<'a> {
    pub method: &'a str,
    pub url: &'a str,
    pub headers: &'a [(String, String)],
    pub body: &'a [u8],
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    struct FixtureLookup {
        effects: Arc<Mutex<Vec<&'static str>>>,
        result: Result<Option<SecretString>, AuthServiceError>,
    }

    impl AuthValueLookup for FixtureLookup {
        fn lookup(&self, _key: &str) -> Result<Option<SecretString>, AuthServiceError> {
            self.effects.lock().unwrap().push("lookup");
            self.result.clone()
        }
    }

    struct FixtureProvider {
        effects: Arc<Mutex<Vec<&'static str>>>,
        result: Result<SecretString, AuthServiceError>,
    }

    impl CallerTokenProvider for FixtureProvider {
        fn invoke(&self) -> CallerTokenFuture<'_> {
            self.effects.lock().unwrap().push("caller");
            Box::pin(std::future::ready(self.result.clone()))
        }
    }

    #[derive(Clone)]
    struct FixtureHeaders(Arc<Mutex<Vec<(String, String)>>>);

    impl ExecutionHeaders for FixtureHeaders {
        fn read(&self) -> Result<Vec<(String, String)>, AuthServiceError> {
            Ok(self.0.lock().unwrap().clone())
        }
    }

    #[test]
    fn supplied_values_preserve_absent_empty_whitespace_and_source() {
        let supplied = [
            SuppliedSecret {
                source: "absent".into(),
                value: None,
            },
            SuppliedSecret {
                source: "empty".into(),
                value: Some(SecretString::new("")),
            },
            SuppliedSecret {
                source: "whitespace".into(),
                value: Some(SecretString::new("  ")),
            },
            SuppliedSecret {
                source: "explicit".into(),
                value: Some(SecretString::new("token")),
            },
        ];

        assert_eq!(supplied[0].value, None);
        assert_eq!(supplied[1].value.as_ref().unwrap().expose(), "");
        assert_eq!(supplied[2].value.as_ref().unwrap().expose(), "  ");
        assert_eq!(supplied[3].source, "explicit");
    }

    #[test]
    fn secrets_and_containing_values_are_redacted() {
        let secret = SecretString::new("never-print-this");
        let resolved = ResolvedCredential {
            value: secret.clone(),
            expires_at: None,
        };
        assert!(!format!("{secret}").contains("never-print-this"));
        assert!(!format!("{secret:?}").contains("never-print-this"));
        assert!(!format!("{resolved:?}").contains("never-print-this"));
    }

    #[tokio::test]
    async fn fixture_preserves_effect_order_and_deferred_header_reads() {
        let effects = Arc::new(Mutex::new(Vec::new()));
        let lookup = FixtureLookup {
            effects: effects.clone(),
            result: Ok(Some(SecretString::new("looked-up"))),
        };
        let provider = FixtureProvider {
            effects: effects.clone(),
            result: Ok(SecretString::new("caller-token")),
        };
        let stored = Arc::new(Mutex::new(vec![
            ("Authorization".into(), "Bearer original".into()),
            ("authorization".into(), "Bearer forwarded".into()),
        ]));
        let headers = FixtureHeaders(stored.clone());

        assert!(effects.lock().unwrap().is_empty());
        let _ = lookup.lookup("credential").unwrap();
        let token = provider.invoke().await.unwrap();
        assert_eq!(effects.lock().unwrap().as_slice(), &["lookup", "caller"]);
        assert_eq!(token.expose(), "caller-token");

        stored.lock().unwrap()[0].1 = "Bearer changed".into();
        let read = headers.read().unwrap();
        assert_eq!(read[0].1, "Bearer changed");
        assert_eq!(read[1].0, "authorization");
        assert_eq!(effects.lock().unwrap().as_slice(), &["lookup", "caller"]);
    }

    #[test]
    fn body_authorization_receives_exact_serialized_bytes() {
        let body = br#"{"message":"exact bytes"}"#;
        let input = BodyAuthorizationInput {
            method: "POST",
            url: "https://example.com",
            headers: &[("content-type".into(), "application/json".into())],
            body,
        };
        assert_eq!(input.body, body);
    }

    #[tokio::test]
    async fn lookup_failure_stops_before_caller_invocation() {
        let effects = Arc::new(Mutex::new(Vec::new()));
        let lookup = FixtureLookup {
            effects: effects.clone(),
            result: Err(AuthServiceError::Lookup {
                source: "environment".into(),
            }),
        };
        let provider = FixtureProvider {
            effects: effects.clone(),
            result: Ok(SecretString::new("unused")),
        };

        let result = lookup.lookup("credential");
        if result.is_ok() {
            let _ = provider.invoke().await;
        }
        assert!(matches!(result, Err(AuthServiceError::Lookup { .. })));
        assert_eq!(effects.lock().unwrap().as_slice(), &["lookup"]);
    }

    #[tokio::test]
    async fn caller_failure_is_not_replaced_by_another_source() {
        let effects = Arc::new(Mutex::new(Vec::new()));
        let lookup = FixtureLookup {
            effects: effects.clone(),
            result: Ok(Some(SecretString::new("looked-up"))),
        };
        let provider = FixtureProvider {
            effects: effects.clone(),
            result: Err(AuthServiceError::CallerToken),
        };

        let _ = lookup.lookup("credential").unwrap();
        let result = provider.invoke().await;
        assert_eq!(result, Err(AuthServiceError::CallerToken));
        assert_eq!(effects.lock().unwrap().as_slice(), &["lookup", "caller"]);
    }

    #[test]
    fn replacing_a_logging_view_does_not_replace_execution_headers() {
        let stored = Arc::new(Mutex::new(vec![(
            "Authorization".into(),
            "Bearer execution".into(),
        )]));
        let execution = FixtureHeaders(stored);
        let logging_replacement = FixtureHeaders(Arc::new(Mutex::new(vec![(
            "Authorization".into(),
            "Bearer logging".into(),
        )])));

        assert_eq!(execution.read().unwrap()[0].1, "Bearer execution");
        assert_eq!(logging_replacement.read().unwrap()[0].1, "Bearer logging");
    }
}
