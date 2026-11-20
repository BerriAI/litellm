//! Google login as google-auth does it, for any Google service: the credential files it reads,
//! application default credentials, the metadata server, and the exceptions it raises, built on
//! Google's Rust SDK with backwards-compat modules in [`legacy`] for what the SDK lacks.

pub(crate) mod adc;
pub(crate) mod bounded_file_source;
pub(crate) mod builders;
pub(crate) mod credentials;
pub(crate) mod environment;
pub(crate) mod failure;
pub(crate) mod legacy;

#[cfg(any(test, feature = "test-support"))]
pub use environment::OnGce;
pub use environment::{EnvironmentFuture, GoogleEnvironment, SystemGoogleEnvironment};
pub use failure::{GoogleLoginFailure, PythonException, UnusableFile};

use litellm_auth_types::{Error, ErrorDetail};
use litellm_secret_value::SecretValue;
use std::sync::Arc;

use crate::{GcpAuthFuture, GcpTokenSource};
use failure::UnsignableKey;

/// A loaded login: its token source, the project its credentials resolved, and whether its token
/// is a service account key's own signature.
pub(crate) struct Login {
    pub(crate) source: Arc<dyn GcpTokenSource>,
    pub(crate) project: Option<String>,
    pub(crate) signs: bool,
}

impl GcpTokenSource for Login {
    fn project_id(&self) -> GcpAuthFuture<'_, String> {
        Box::pin(async {
            self.project
                .clone()
                .ok_or(Error::CredentialAcquisition(ErrorDetail::Unresolved {
                    subject: "Vertex AI project",
                }))
        })
    }

    fn token(&self) -> GcpAuthFuture<'_, SecretValue> {
        Box::pin(async {
            self.source.token().await.map_err(|error| match self.signs {
                true => Error::provider(UnsignableKey(error)),
                false => error,
            })
        })
    }
}
