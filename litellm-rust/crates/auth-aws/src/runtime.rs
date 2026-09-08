use std::future::Future;
use std::pin::Pin;
use std::time::SystemTime;

use aws_credential_types::provider::ProvideCredentials;
use veil::Redact;

use crate::{Credentials, Error};

#[derive(Clone, PartialEq, Eq, Redact)]
pub struct AssumeRoleRequest {
    pub role: String,
    pub session_name: String,
    pub region: Option<String>,
    pub endpoint: Option<String>,
    pub source_credentials: Option<Credentials>,
    #[redact(fixed = 8)]
    pub external_id: Option<String>,
}

#[derive(Clone, PartialEq, Eq, Redact)]
pub struct WebIdentityRequest {
    #[redact(fixed = 8)]
    pub token: String,
    pub role: String,
    pub session_name: String,
    pub region: Option<String>,
    pub endpoint: Option<String>,
}

pub type CredentialFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, Error>> + Send + 'a>>;

pub trait CredentialRuntime: Send + Sync {
    fn profile<'a>(&'a self, name: &'a str) -> CredentialFuture<'a, Credentials>;
    fn ambient(&self) -> CredentialFuture<'_, Credentials>;
    fn assume_role(&self, request: AssumeRoleRequest) -> CredentialFuture<'_, Credentials>;
    fn web_identity(&self, request: WebIdentityRequest) -> CredentialFuture<'_, Credentials>;
    fn caller_identity(
        &self,
        region: Option<String>,
        endpoint: Option<String>,
    ) -> CredentialFuture<'_, Option<String>>;
}

#[derive(Default)]
pub struct NativeCredentialRuntime;

impl CredentialRuntime for NativeCredentialRuntime {
    fn profile<'a>(&'a self, name: &'a str) -> CredentialFuture<'a, Credentials> {
        Box::pin(profile_credentials(name))
    }

    fn ambient(&self) -> CredentialFuture<'_, Credentials> {
        Box::pin(default_credentials())
    }

    fn assume_role(&self, request: AssumeRoleRequest) -> CredentialFuture<'_, Credentials> {
        Box::pin(assume_role_credentials(request))
    }

    fn web_identity(&self, request: WebIdentityRequest) -> CredentialFuture<'_, Credentials> {
        Box::pin(web_identity_credentials(request))
    }

    fn caller_identity(
        &self,
        region: Option<String>,
        endpoint: Option<String>,
    ) -> CredentialFuture<'_, Option<String>> {
        Box::pin(caller_identity(region, endpoint))
    }
}

pub async fn profile_credentials(name: &str) -> Result<Credentials, Error> {
    let provider = aws_config::profile::ProfileFileCredentialsProvider::builder()
        .profile_name(name)
        .build();
    provider
        .provide_credentials()
        .await
        .map(Credentials)
        .map_err(|error| Error::ProfileCredentials(error.to_string()))
}

pub async fn default_credentials() -> Result<Credentials, Error> {
    let provider = aws_config::default_provider::credentials::DefaultCredentialsChain::builder()
        .build()
        .await;
    provider
        .provide_credentials()
        .await
        .map(Credentials)
        .map_err(|error| Error::DefaultCredentials(error.to_string()))
}

fn sdk_loader(region: Option<String>, endpoint: Option<String>) -> aws_config::ConfigLoader {
    let loader = aws_config::defaults(aws_config::BehaviorVersion::latest());
    let loader = match region {
        Some(region) => loader.region(aws_types::region::Region::new(region)),
        None => loader,
    };
    match endpoint {
        Some(endpoint) => loader.endpoint_url(endpoint),
        None => loader,
    }
}

pub async fn assume_role_credentials(request: AssumeRoleRequest) -> Result<Credentials, Error> {
    let mut loader = sdk_loader(request.region, request.endpoint);
    if let Some(credentials) = request.source_credentials {
        loader = loader.credentials_provider(credentials.0);
    }
    let sdk_config = loader.load().await;
    let builder = aws_config::sts::AssumeRoleProvider::builder(request.role)
        .session_name(request.session_name);
    let builder = match request.external_id {
        Some(id) => builder.external_id(id),
        None => builder,
    };
    builder
        .configure(&sdk_config)
        .build()
        .await
        .provide_credentials()
        .await
        .map(Credentials)
        .map_err(|error| Error::RoleCredentials(error.to_string()))
}

pub async fn web_identity_credentials(request: WebIdentityRequest) -> Result<Credentials, Error> {
    let sdk_config = sdk_loader(request.region, request.endpoint).load().await;
    let response = aws_sdk_sts::Client::new(&sdk_config)
        .assume_role_with_web_identity()
        .role_arn(request.role)
        .role_session_name(request.session_name)
        .web_identity_token(request.token)
        .send()
        .await
        .map_err(|error| Error::WebIdentityCredentials(error.to_string()))?;
    let credentials = response
        .credentials()
        .ok_or(Error::MissingWebIdentityCredentials)?;
    let expiration = SystemTime::try_from(*credentials.expiration())
        .map_err(|error| Error::InvalidWebIdentityExpiration(error.to_string()))?;
    Ok(Credentials::new(
        credentials.access_key_id(),
        credentials.secret_access_key(),
        Some(credentials.session_token().to_string()),
        Some(expiration),
        "litellm-web-identity",
    ))
}

pub async fn caller_identity(
    region: Option<String>,
    endpoint: Option<String>,
) -> Result<Option<String>, Error> {
    let sdk_config = sdk_loader(region, endpoint).load().await;
    match aws_sdk_sts::Client::new(&sdk_config)
        .get_caller_identity()
        .send()
        .await
    {
        Ok(response) => Ok(response.arn().map(str::to_string)),
        Err(_) => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::static_credentials;

    struct FixtureRuntime {
        effects: Arc<Mutex<Vec<&'static str>>>,
    }

    impl CredentialRuntime for FixtureRuntime {
        fn profile<'a>(&'a self, _name: &'a str) -> CredentialFuture<'a, Credentials> {
            self.effects.lock().unwrap().push("profile");
            Box::pin(std::future::ready(Ok(static_credentials(
                "profile", "secret",
            ))))
        }

        fn ambient(&self) -> CredentialFuture<'_, Credentials> {
            self.effects.lock().unwrap().push("ambient");
            Box::pin(std::future::ready(Ok(static_credentials(
                "ambient", "secret",
            ))))
        }

        fn assume_role(&self, _request: AssumeRoleRequest) -> CredentialFuture<'_, Credentials> {
            self.effects.lock().unwrap().push("assume-role");
            Box::pin(std::future::ready(Ok(static_credentials("role", "secret"))))
        }

        fn web_identity(&self, _request: WebIdentityRequest) -> CredentialFuture<'_, Credentials> {
            self.effects.lock().unwrap().push("web-identity");
            Box::pin(std::future::ready(Ok(static_credentials("web", "secret"))))
        }

        fn caller_identity(
            &self,
            _region: Option<String>,
            _endpoint: Option<String>,
        ) -> CredentialFuture<'_, Option<String>> {
            self.effects.lock().unwrap().push("caller-identity");
            Box::pin(std::future::ready(Ok(None)))
        }
    }

    #[test]
    fn mechanism_inputs_are_redacted() {
        let assume_role = AssumeRoleRequest {
            role: "role".into(),
            session_name: "session".into(),
            region: None,
            endpoint: None,
            source_credentials: Some(static_credentials("visible-id", "secret")),
            external_id: Some("external-secret".into()),
        };
        let web_identity = WebIdentityRequest {
            token: "identity-secret".into(),
            role: "role".into(),
            session_name: "session".into(),
            region: None,
            endpoint: None,
        };
        let debug = format!("{assume_role:?} {web_identity:?}");
        for secret in ["visible-id", "secret", "external-secret", "identity-secret"] {
            assert!(!debug.contains(secret));
        }
    }

    #[tokio::test]
    async fn credential_io_is_injectable_for_policy_consumers() {
        let effects = Arc::new(Mutex::new(Vec::new()));
        let runtime = FixtureRuntime {
            effects: effects.clone(),
        };

        assert_eq!(
            runtime.profile("demo").await.unwrap().access_key_id(),
            "profile"
        );
        assert_eq!(runtime.ambient().await.unwrap().access_key_id(), "ambient");
        assert_eq!(
            runtime
                .caller_identity(Some("us-east-1".into()), None)
                .await
                .unwrap(),
            None
        );
        assert_eq!(
            effects.lock().unwrap().as_slice(),
            &["profile", "ambient", "caller-identity"]
        );
    }
}
