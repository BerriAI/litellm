//! The credentials a Vertex AI Messages login holds, as google-auth's `Credentials`: something
//! that mints a new token on `refresh`, and the token it minted last with its expiry.

use std::{future::Future, pin::Pin, sync::Arc, time::Duration};

use litellm_secret_value::SecretValue;
use serde::Deserialize;
use time::OffsetDateTime;

use super::{
    builders::{TokenSource, token_failure},
    failure::{GoogleLoginFailure, PythonException},
    source::present,
};

/// `google.auth._helpers.REFRESH_THRESHOLD`: a token this close to its expiry is stale.
pub(crate) const REFRESH_THRESHOLD: Duration = Duration::from_secs(3 * 60 + 45);

pub(crate) type RefreshFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Minted, GoogleLoginFailure>> + Send + 'a>>;

/// google-auth's `Credentials.refresh`: a new token, whatever the last one was.
pub(crate) trait Credentials: Send + Sync {
    fn refresh(&self) -> RefreshFuture<'_>;
}

pub(crate) type Loaded = Arc<dyn Credentials>;

/// When a token stops being usable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Expiry {
    At(OffsetDateTime),
    /// google-auth treats a token without an expiry as fresh for good.
    Never,
    /// Google's Rust SDK keeps its own token and refreshes it ahead of expiry, so the login asks
    /// it every time instead of holding a token.
    Managed,
}

/// A token a refresh minted: `None` when the token endpoint's reply named its access token null,
/// which google-auth keeps as a token it can never use.
#[derive(Clone, Debug)]
pub(crate) struct Minted {
    pub(crate) token: Option<SecretValue>,
    pub(crate) expiry: Expiry,
}

/// google-auth's `TokenState`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TokenState {
    Fresh,
    Stale,
    Invalid,
}

/// A token endpoint's `expires_in`: seconds, or the seconds as text.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub(crate) enum Lifetime {
    Seconds(f64),
    Text(String),
}

/// A token endpoint's JSON reply to a grant.
#[derive(Clone, Default, Deserialize, PartialEq, veil::Redact)]
pub(crate) struct TokenReply {
    /// `None` when the reply has no `access_token`, `Some(None)` when it is null.
    #[redact(with = "[REDACTED]")]
    #[serde(default, deserialize_with = "present")]
    pub(crate) access_token: Option<Option<String>>,
    pub(crate) expires_in: Option<Lifetime>,
    #[redact(with = "[REDACTED]")]
    pub(crate) refresh_token: Option<String>,
}

impl TokenReply {
    /// The reply a token endpoint answered a grant with, which must be a JSON object.
    pub(crate) fn parse(body: &str) -> Result<Self, GoogleLoginFailure> {
        serde_json::from_str(body).map_err(|_| not_a_token_reply())
    }
}

/// A token endpoint's JSON reply to a refused grant.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub(crate) struct TokenRefusal {
    pub(crate) error: Option<String>,
    pub(crate) error_description: Option<String>,
    pub(crate) error_subtype: Option<String>,
    pub(crate) error_uri: Option<String>,
}

impl TokenRefusal {
    /// The refusal a body holds, when it is a JSON object of text fields.
    pub(crate) fn parse(body: &str) -> Option<Self> {
        serde_json::from_str(body).ok()
    }
}

/// A token reply that is not a JSON object of the fields a grant answers with.
pub(crate) fn not_a_token_reply() -> GoogleLoginFailure {
    GoogleLoginFailure::new(
        PythonException::Refresh,
        "The token endpoint's reply is not a JSON object.",
    )
}

impl Minted {
    pub(crate) fn new(value: SecretValue, expiry: Expiry) -> Self {
        Self {
            token: Some(value),
            expiry,
        }
    }

    /// The token google-auth sets from a reply's `access_token`.
    pub(crate) fn from_reply(token: Option<String>, expiry: Expiry) -> Self {
        Self {
            token: token.map(SecretValue::new),
            expiry,
        }
    }

    /// The final validation of `get_access_token_async`.
    pub(crate) fn usable(self) -> Result<SecretValue, GoogleLoginFailure> {
        self.token.ok_or_else(|| {
            GoogleLoginFailure::new(
                PythonException::Value,
                "Could not resolve credentials token. Got None or non-string token (type=NoneType)",
            )
        })
    }

    pub(crate) fn state(&self, now: OffsetDateTime) -> TokenState {
        if self.token.is_none() {
            return TokenState::Invalid;
        }
        match self.expiry {
            Expiry::Never => TokenState::Fresh,
            Expiry::Managed => TokenState::Invalid,
            Expiry::At(expiry) if now >= expiry => TokenState::Invalid,
            Expiry::At(expiry) if now >= expiry - REFRESH_THRESHOLD => TokenState::Stale,
            Expiry::At(_) => TokenState::Fresh,
        }
    }
}

/// `VertexBase._refresh_rejected`: Google refused the credentials for good, so the login loads
/// them again once, which picks up a re-run `gcloud auth application-default login` or a rotated
/// key.
pub(crate) fn refresh_rejected(failure: &GoogleLoginFailure) -> bool {
    failure.grant_rejected
}

impl TokenRefusal {
    /// Google's `invalid_grant`: the refresh token or key is revoked, expired or wrong.
    pub(crate) fn rejects_the_grant(&self) -> bool {
        self.error.as_deref() == Some("invalid_grant")
    }
}

/// A token source from Google's Rust SDK, which keeps and refreshes its own token.
struct Managed(TokenSource);

impl Credentials for Managed {
    fn refresh(&self) -> RefreshFuture<'_> {
        Box::pin(async {
            let value = self
                .0
                .token()
                .await
                .map_err(|error| token_failure(&error))?;
            Ok(Minted::new(value, Expiry::Managed))
        })
    }
}

pub(crate) fn managed(source: TokenSource) -> Loaded {
    Arc::new(Managed(source))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use time::macros::datetime;

    use super::*;

    const NOW: OffsetDateTime = datetime!(2026-11-16 12:00 UTC);

    #[rstest]
    fn token_reply_debug_never_shows_a_token() {
        let reply: TokenReply = serde_json::from_value(serde_json::json!({
            "access_token": "credential-value",
            "refresh_token": "credential-value",
        }))
        .unwrap();
        let debug = format!("{reply:?}");
        assert!(!debug.contains("credential-value"), "{debug}");
        assert!(debug.contains("[REDACTED]"), "{debug}");
    }

    fn minted(expiry: Expiry) -> Minted {
        Minted::new(SecretValue::new("t"), expiry)
    }

    #[rstest]
    #[case::well_before_expiry(Expiry::At(NOW + Duration::from_secs(226)), TokenState::Fresh)]
    #[case::inside_the_threshold(Expiry::At(NOW + Duration::from_secs(225)), TokenState::Stale)]
    #[case::just_before_expiry(Expiry::At(NOW + Duration::from_secs(1)), TokenState::Stale)]
    #[case::at_expiry(Expiry::At(NOW), TokenState::Invalid)]
    #[case::no_expiry(Expiry::Never, TokenState::Fresh)]
    #[case::sdk_managed(Expiry::Managed, TokenState::Invalid)]
    fn a_token_s_state_follows_google_auth_s_thresholds(
        #[case] expiry: Expiry,
        #[case] expected: TokenState,
    ) {
        assert_eq!(minted(expiry).state(NOW), expected);
    }

    #[rstest]
    #[case::null(None, true)]
    #[case::text(Some("t"), false)]
    fn a_null_token_is_never_usable(#[case] token: Option<&str>, #[case] invalid: bool) {
        let minted = Minted::from_reply(token.map(str::to_owned), Expiry::Never);
        assert_eq!(minted.state(NOW) == TokenState::Invalid, invalid);
        let usable = minted.usable().map(|token| token.expose().to_owned());
        match token {
            Some(token) => assert_eq!(usable, Ok(token.to_owned())),
            None => assert_eq!(usable.unwrap_err().exception, PythonException::Value),
        }
    }

    #[rstest]
    #[case::granted(
        r#"{"access_token": "t", "expires_in": 60}"#,
        Some(Some("t")),
        Some(Lifetime::Seconds(60.0))
    )]
    #[case::null_token(r#"{"access_token": null, "expires_in": "60"}"#, Some(None), Some(Lifetime::Text("60".into())))]
    #[case::no_token(r#"{"token_type": "Bearer", "scope": ["a", 1]}"#, None, None)]
    fn a_token_reply_tells_a_null_token_from_a_missing_one(
        #[case] body: &str,
        #[case] token: Option<Option<&str>>,
        #[case] lifetime: Option<Lifetime>,
    ) {
        let reply = TokenReply::parse(body).unwrap();
        assert_eq!(
            reply.access_token,
            token.map(|token| token.map(str::to_owned))
        );
        assert_eq!(reply.expires_in, lifetime);
    }

    #[rstest]
    #[case::not_json("nope")]
    #[case::list("[1]")]
    #[case::numeric_token(r#"{"access_token": 7}"#)]
    #[case::boolean_lifetime(r#"{"access_token": "t", "expires_in": true}"#)]
    fn a_reply_that_is_not_a_token_reply_is_refused(#[case] body: &str) {
        assert_eq!(TokenReply::parse(body).unwrap_err(), not_a_token_reply());
    }

    #[rstest]
    #[case::invalid_grant(
        r#"{"error": "invalid_grant", "error_description": "Invalid JWT Signature."}"#,
        true
    )]
    #[case::another_refusal(
        r#"{"error": "invalid_client", "error_description": "invalid_grant"}"#,
        false
    )]
    #[case::text_naming_invalid_grant(r#""invalid_grant""#, false)]
    #[case::not_json("invalid_grant", false)]
    fn only_an_invalid_grant_error_reloads_the_credentials(
        #[case] body: &str,
        #[case] expected: bool,
    ) {
        let failure = super::super::service_account::refused(body);
        assert_eq!(failure.exception, PythonException::Refresh);
        assert_eq!(refresh_rejected(&failure), expected);
    }
}
