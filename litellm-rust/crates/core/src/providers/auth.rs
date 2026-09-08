use std::sync::{Arc, OnceLock};
use std::time::SystemTime;

#[cfg(feature = "bedrock-auth")]
use super::bedrock::aws_base::{
    AwsAuthConfig, AwsCredentialFuture, MAX_CACHED_CREDENTIALS, resolve_credentials_with_state,
};

#[cfg(feature = "bedrock-auth")]
pub type AwsCredentialState = litellm_auth_aws::CredentialState<
    Arc<dyn litellm_auth_aws::CredentialRuntime>,
    Arc<dyn litellm_auth_aws::Clock>,
>;

#[cfg(feature = "bedrock-auth")]
pub(crate) fn native_aws_credential_state() -> AwsCredentialState {
    AwsCredentialState::with_clock(
        Arc::new(litellm_auth_aws::NativeCredentialRuntime),
        MAX_CACHED_CREDENTIALS,
        Arc::new(litellm_auth_aws::SystemClock),
    )
}

pub trait Environment: Send + Sync {
    fn environment(&self, key: &str) -> Option<String>;
}

pub trait SigningClock: Send + Sync {
    fn signing_time(&self) -> SystemTime;
}

pub trait AwsMechanisms: Send + Sync {
    #[cfg(feature = "bedrock-auth")]
    fn aws_credential_state(&self) -> &AwsCredentialState;
}

pub trait ChatAuthorizationServices: Environment + SigningClock + AwsMechanisms {}

impl<T> ChatAuthorizationServices for T where T: Environment + SigningClock + AwsMechanisms {}

pub trait AuthorizationServices: Send + Sync {
    fn environment(&self, key: &str) -> Option<String>;

    fn signing_time(&self) -> SystemTime;

    #[cfg(feature = "bedrock-auth")]
    fn resolve_aws_credentials<'a>(&'a self, config: AwsAuthConfig) -> AwsCredentialFuture<'a>;
}

pub struct NativeAuthorizationServices {
    #[cfg(feature = "bedrock-auth")]
    aws_credentials: AwsCredentialState,
}

impl NativeAuthorizationServices {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "bedrock-auth")]
            aws_credentials: native_aws_credential_state(),
        }
    }
}

impl Default for NativeAuthorizationServices {
    fn default() -> Self {
        Self::new()
    }
}

impl AuthorizationServices for NativeAuthorizationServices {
    fn environment(&self, key: &str) -> Option<String> {
        Environment::environment(self, key)
    }

    fn signing_time(&self) -> SystemTime {
        SigningClock::signing_time(self)
    }

    #[cfg(feature = "bedrock-auth")]
    fn resolve_aws_credentials<'a>(&'a self, config: AwsAuthConfig) -> AwsCredentialFuture<'a> {
        Box::pin(async move {
            let environment = |key: &str| Environment::environment(self, key);
            resolve_credentials_with_state(config, &environment, &self.aws_credentials).await
        })
    }
}

impl Environment for NativeAuthorizationServices {
    fn environment(&self, key: &str) -> Option<String> {
        std::env::var(key).ok()
    }
}

impl SigningClock for NativeAuthorizationServices {
    fn signing_time(&self) -> SystemTime {
        SystemTime::now()
    }
}

impl AwsMechanisms for NativeAuthorizationServices {
    #[cfg(feature = "bedrock-auth")]
    fn aws_credential_state(&self) -> &AwsCredentialState {
        &self.aws_credentials
    }
}

pub fn native_authorization_services() -> &'static NativeAuthorizationServices {
    shared_native_authorization_services()
}

pub fn shared_native_authorization_services() -> &'static Arc<NativeAuthorizationServices> {
    static SERVICES: OnceLock<Arc<NativeAuthorizationServices>> = OnceLock::new();
    SERVICES.get_or_init(|| Arc::new(NativeAuthorizationServices::new()))
}
