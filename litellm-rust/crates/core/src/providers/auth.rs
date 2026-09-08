use std::sync::{Arc, OnceLock};
use std::time::SystemTime;

#[cfg(feature = "bedrock-auth")]
use super::bedrock::aws_base::{
    AwsAuthConfig, AwsCredentialFuture, AwsCredentialService, NativeAwsCredentialService,
};

pub trait AuthorizationServices: Send + Sync {
    fn environment(&self, key: &str) -> Option<String>;

    fn signing_time(&self) -> SystemTime;

    #[cfg(feature = "bedrock-auth")]
    fn resolve_aws_credentials<'a>(&'a self, config: AwsAuthConfig) -> AwsCredentialFuture<'a>;
}

pub struct NativeAuthorizationServices {
    #[cfg(feature = "bedrock-auth")]
    aws_credentials: NativeAwsCredentialService,
}

impl NativeAuthorizationServices {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "bedrock-auth")]
            aws_credentials: NativeAwsCredentialService::new(),
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
        std::env::var(key).ok()
    }

    fn signing_time(&self) -> SystemTime {
        SystemTime::now()
    }

    #[cfg(feature = "bedrock-auth")]
    fn resolve_aws_credentials<'a>(&'a self, config: AwsAuthConfig) -> AwsCredentialFuture<'a> {
        Box::pin(async move {
            let environment = |key: &str| self.environment(key);
            self.aws_credentials.resolve(config, &environment).await
        })
    }
}

pub fn native_authorization_services() -> &'static NativeAuthorizationServices {
    shared_native_authorization_services()
}

pub fn shared_native_authorization_services() -> &'static Arc<NativeAuthorizationServices> {
    static SERVICES: OnceLock<Arc<NativeAuthorizationServices>> = OnceLock::new();
    SERVICES.get_or_init(|| Arc::new(NativeAuthorizationServices::new()))
}
