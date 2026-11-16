//! The credentials a Vertex AI Messages login holds, as google-auth's `Credentials`: something
//! that mints a new token on `refresh`, and the token it minted last with its expiry.

use std::{future::Future, pin::Pin, sync::Arc, time::Duration};

use litellm_secret_value::SecretValue;
use time::OffsetDateTime;

use super::{
    builders::{TokenSource, token_failure},
    failure::{GoogleLoginFailure, PythonException},
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

#[derive(Clone, Debug)]
pub(crate) struct Minted {
    pub(crate) value: SecretValue,
    pub(crate) expiry: Expiry,
}

/// google-auth's `TokenState`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TokenState {
    Fresh,
    Stale,
    Invalid,
}

impl Minted {
    pub(crate) fn state(&self, now: OffsetDateTime) -> TokenState {
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
    (failure.exception == PythonException::Refresh && failure.message.contains("invalid_grant"))
        || failure.message.contains("Reauthentication is needed")
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
            Ok(Minted {
                value,
                expiry: Expiry::Managed,
            })
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

    fn minted(expiry: Expiry) -> Minted {
        Minted {
            value: SecretValue::new("t"),
            expiry,
        }
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
    #[case::invalid_grant(
        PythonException::Refresh,
        "('invalid_grant: Invalid JWT Signature.', {'error': 'invalid_grant'})",
        true
    )]
    #[case::reauthentication(
        PythonException::Refresh,
        "Reauthentication is needed. Please run `gcloud auth application-default login`.",
        true
    )]
    #[case::another_refusal(
        PythonException::Refresh,
        "('invalid_client: The OAuth client was not found.', {'error': 'invalid_client'})",
        false
    )]
    #[case::unreachable(PythonException::Transport, "invalid_grant", false)]
    fn only_a_refusal_for_good_reloads_the_credentials(
        #[case] exception: PythonException,
        #[case] message: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(
            refresh_rejected(&GoogleLoginFailure::new(exception, message)),
            expected
        );
    }
}
