//! v0 [`KeyAuthenticator`] backed by the Python proxy.
//!
//! Verification is a single POST to the proxy's internal auth endpoint, authorized
//! by a shared data-plane key. The HTTP client is built once and reused (pooled
//! connections + TLS), with bounded connect and request timeouts so a slow proxy
//! can never hang a request indefinitely. TLS is rustls for portable builds.

use std::time::Duration;

use reqwest::StatusCode;

use crate::auth::{AuthError, KeyAuthenticator, UserApiKeyAuth};

/// Bound how long we wait to establish a connection to the proxy.
const CONNECT_TIMEOUT_SECS: u64 = 5;
/// Bound the full verify round-trip. Auth must be fast; a slow proxy fails closed.
const REQUEST_TIMEOUT_SECS: u64 = 10;
/// Header carrying the shared data-plane secret that authorizes this gateway to
/// the proxy's internal auth endpoint.
const DATA_PLANE_KEY_HEADER: &str = "X-LiteLLM-Data-Plane-Key";

/// Calls the Python proxy to verify virtual keys. One pooled client, reused.
pub struct PythonAuthClient {
    http: reqwest::Client,
    verify_url: String,
    data_plane_key: String,
}

impl PythonAuthClient {
    /// Build the client with bounded timeouts and rustls TLS. Panics only if the
    /// reqwest builder fails, which is a startup misconfiguration, not a runtime
    /// path — acceptable to surface loudly at boot.
    pub fn new(verify_url: String, data_plane_key: String) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(CONNECT_TIMEOUT_SECS))
            .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
            .use_rustls_tls()
            .build()
            .expect("failed to build auth HTTP client");
        Self {
            http,
            verify_url,
            data_plane_key,
        }
    }
}

#[axum::async_trait]
impl KeyAuthenticator for PythonAuthClient {
    async fn verify(&self, key: &str, route: &str) -> Result<UserApiKeyAuth, AuthError> {
        let response = self
            .http
            .post(&self.verify_url)
            .header(DATA_PLANE_KEY_HEADER, &self.data_plane_key)
            .json(&serde_json::json!({ "api_key": key, "route": route }))
            .send()
            .await
            .map_err(|err| AuthError::Upstream(err.to_string()))?;

        let status = response.status();
        if status == StatusCode::UNAUTHORIZED {
            return Err(AuthError::Unauthorized);
        }
        if !status.is_success() {
            return Err(AuthError::Upstream(format!(
                "auth verify returned status {}",
                status.as_u16()
            )));
        }

        response
            .json::<UserApiKeyAuth>()
            .await
            .map_err(|err| AuthError::Upstream(format!("invalid auth response: {err}")))
    }
}
