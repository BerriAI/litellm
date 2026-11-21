//! Backwards compatibility only, with google-auth impersonating a service account at the
//! `service_account_impersonation_url` an external account names. Google's Rust SDK
//! (`google_cloud_auth` 1.16) impersonates credentials it is handed only through the public
//! `iamcredentials.googleapis.com`, so an account naming a Private Service Connect host or another
//! universe would be sent elsewhere. For such a URL this module makes `impersonated_credentials`'
//! `generateAccessToken` call itself, with the source credentials' token as the bearer. Unlike
//! google-auth, it refuses a plain `http` URL unless it names this machine, so the bearer never
//! crosses a network in the clear. Delete it once the SDK honours a custom impersonation URL for
//! credentials it is handed.

use std::sync::Arc;

use litellm_secret_value::SecretValue;
use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use super::token::{CompatLogin, Granted, HeldToken};
use crate::{
    GcpTokenSource,
    constants::CLOUD_PLATFORM_SCOPE,
    google::{
        environment::{GoogleEnvironment, unreachable},
        failure::{GoogleLoginFailure, PythonException},
    },
};

const SDK_IMPERSONATION_HOST: &str = "iamcredentials.googleapis.com";
/// `impersonated_credentials._DEFAULT_TOKEN_LIFETIME_SECS`.
const DEFAULT_LIFETIME_SECONDS: u64 = 3600;
const REFUSED: &str = "Vertex: the service account impersonation endpoint refused the request.";
const NO_TOKEN: &str = "Vertex: the service account impersonation endpoint answered no token.";
const PLAIN_HTTP: &str =
    "Vertex: the service account impersonation URL must use https unless it names this machine.";

/// Whether Google's Rust SDK would call `url` itself: an HTTPS URL on its default host.
pub(super) fn sdk_calls(url: &str) -> bool {
    url::Url::parse(url)
        .is_ok_and(|url| url.scheme() == "https" && url.host_str() == Some(SDK_IMPERSONATION_HOST))
}

/// Whether `url` is plain `http` to a host other than this machine's loopback, which would send
/// the source credentials' token over the network unencrypted.
fn plain_http_off_this_machine(url: &str) -> bool {
    url::Url::parse(url).is_ok_and(|url| {
        url.scheme() == "http"
            && !match url.host() {
                Some(url::Host::Domain(domain)) => domain == "localhost",
                Some(url::Host::Ipv4(address)) => address.is_loopback(),
                Some(url::Host::Ipv6(address)) => address.is_loopback(),
                None => false,
            }
    })
}

/// `_make_iam_token_request`'s body.
#[derive(Serialize)]
struct GenerateAccessToken<'a> {
    delegates: Option<Vec<String>>,
    scope: [&'a str; 1],
    lifetime: String,
}

/// `generateAccessToken`'s answer.
#[derive(Deserialize, veil::Redact)]
#[serde(rename_all = "camelCase")]
struct Generated {
    #[redact(with = "[REDACTED]")]
    access_token: String,
    expire_time: String,
}

/// A service account impersonated at a URL Google's Rust SDK would not call, with `source`'s
/// token.
pub(super) struct CustomImpersonation {
    source: Arc<dyn GcpTokenSource>,
    url: String,
    lifetime_seconds: u64,
    held: HeldToken,
    environment: Arc<dyn GoogleEnvironment>,
}

impl CustomImpersonation {
    pub(super) fn new(
        source: Arc<dyn GcpTokenSource>,
        url: &str,
        lifetime_seconds: Option<u64>,
        environment: Arc<dyn GoogleEnvironment>,
    ) -> Self {
        Self {
            source,
            url: url.to_owned(),
            lifetime_seconds: lifetime_seconds.unwrap_or(DEFAULT_LIFETIME_SECONDS),
            held: HeldToken::default(),
            environment,
        }
    }

    /// `impersonated_credentials.Credentials._update_token`: one `generateAccessToken` call, a
    /// `RefreshError` when it is refused or answers no token, or when the URL is plain `http` to
    /// another machine.
    async fn generate(&self) -> Result<Granted, GoogleLoginFailure> {
        if plain_http_off_this_machine(&self.url) {
            return Err(GoogleLoginFailure::new(
                PythonException::Refresh,
                PLAIN_HTTP,
            ));
        }
        let bearer = self
            .source
            .token()
            .await
            .map_err(|error| GoogleLoginFailure::token(&error, &*self.environment))?;
        let response = self
            .environment
            .http_client(&self.url, None)?
            .post(&self.url)
            .bearer_auth(bearer.expose())
            .json(&GenerateAccessToken {
                delegates: None,
                scope: [CLOUD_PLATFORM_SCOPE],
                lifetime: format!("{}s", self.lifetime_seconds),
            })
            .send()
            .await
            .map_err(unreachable)?;
        if !response.status().is_success() {
            return Err(GoogleLoginFailure::new(PythonException::Refresh, REFUSED));
        }
        let generated: Generated = response
            .json()
            .await
            .map_err(|_| GoogleLoginFailure::new(PythonException::Refresh, NO_TOKEN))?;
        let expire_time = OffsetDateTime::parse(
            &generated.expire_time,
            &time::format_description::well_known::Rfc3339,
        )
        .map_err(|_| GoogleLoginFailure::new(PythonException::Refresh, NO_TOKEN))?;
        let seconds = (expire_time - OffsetDateTime::now_utc()).whole_seconds();
        Ok(Granted {
            access_token: generated.access_token,
            expires_in: Some(u64::try_from(seconds).unwrap_or_default()),
            refresh_token: None,
        })
    }
}

impl CompatLogin for CustomImpersonation {
    async fn token(&self) -> Result<SecretValue, GoogleLoginFailure> {
        self.held.get_or_fetch(self.generate()).await
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::sdk_host(
        "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/sa:generateAccessToken",
        true
    )]
    #[case::private_service_connect(
        "https://iamcredentials-psc.p.googleapis.com/v1/projects/-/serviceAccounts/sa:generateAccessToken",
        false
    )]
    #[case::other_universe(
        "https://iamcredentials.example-universe.com/v1/projects/-/serviceAccounts/sa:generateAccessToken",
        false
    )]
    #[case::plain_http(
        "http://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/sa:generateAccessToken",
        false
    )]
    #[case::not_a_url("iamcredentials.googleapis.com", false)]
    fn only_the_sdk_s_own_host_is_left_to_the_sdk(#[case] url: &str, #[case] expected: bool) {
        assert_eq!(sdk_calls(url), expected);
    }

    #[rstest]
    #[case::https(
        "https://iamcredentials-psc.p.googleapis.com/v1/projects/-/serviceAccounts/sa:generateAccessToken",
        false
    )]
    #[case::loopback_ipv4(
        "http://127.0.0.1:8080/v1/projects/-/serviceAccounts/sa:generateAccessToken",
        false
    )]
    #[case::loopback_ipv4_range(
        "http://127.0.0.2/v1/projects/-/serviceAccounts/sa:generateAccessToken",
        false
    )]
    #[case::loopback_ipv6(
        "http://[::1]:8080/v1/projects/-/serviceAccounts/sa:generateAccessToken",
        false
    )]
    #[case::localhost(
        "http://LOCALHOST:8080/v1/projects/-/serviceAccounts/sa:generateAccessToken",
        false
    )]
    #[case::google_host(
        "http://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/sa:generateAccessToken",
        true
    )]
    #[case::private_address(
        "http://10.0.0.5/v1/projects/-/serviceAccounts/sa:generateAccessToken",
        true
    )]
    #[case::localhost_lookalike(
        "http://localhost.example.com/v1/projects/-/serviceAccounts/sa:generateAccessToken",
        true
    )]
    #[case::not_a_url("iamcredentials.googleapis.com", false)]
    fn plain_http_is_refused_unless_it_stays_on_this_machine(
        #[case] url: &str,
        #[case] refused: bool,
    ) {
        assert_eq!(plain_http_off_this_machine(url), refused);
    }
}
