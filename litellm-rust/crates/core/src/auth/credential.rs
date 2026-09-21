use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use veil::Redact;

use crate::AuthError;

use super::{ResolvedCredential, SecretValue, TokenProviderHandle};

pub fn credential_index(requested: &str, names: &[String]) -> Option<usize> {
    names.iter().position(|name| name == requested)
}

pub fn credential_default_fields<'a>(
    supplied: &[String],
    credential_fields: &'a [String],
) -> Vec<&'a str> {
    credential_fields
        .iter()
        .filter(|name| !supplied.contains(name))
        .map(String::as_str)
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CredentialFileRef {
    Path(PathBuf),
    EnvironmentVariable(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CredentialRef {
    Explicit(SecretValue),
    Env(String),
    File(CredentialFileRef),
    Request(String),
    Host(String),
    None,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CredentialLookup {
    Found(SecretValue),
    Missing,
    Declined,
}

pub type CredentialLookupFuture<'a> =
    Pin<Box<dyn Future<Output = Result<CredentialLookup, AuthError>> + Send + 'a>>;

pub trait CredentialResolver: std::fmt::Debug + Send + Sync {
    fn resolve<'a>(&'a self, reference: &'a CredentialRef) -> CredentialLookupFuture<'a>;
}

#[derive(Clone, Redact)]
pub struct CredentialResolverHandle(#[redact(with = "[REDACTED]")] Arc<dyn CredentialResolver>);

impl CredentialResolverHandle {
    pub fn new(resolver: Arc<dyn CredentialResolver>) -> Self {
        Self(resolver)
    }

    pub async fn resolve(&self, reference: &CredentialRef) -> Result<CredentialLookup, AuthError> {
        self.0.resolve(reference).await
    }
}

#[derive(Clone, Debug)]
pub enum CredentialPlan {
    Static(CredentialRef),
    Caller(TokenProviderHandle),
    None,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CredentialPlanResolution {
    Resolved(ResolvedCredential),
    Unavailable,
}

impl CredentialPlan {
    pub async fn resolve(
        &self,
        resolver: &CredentialResolverHandle,
    ) -> Result<CredentialPlanResolution, AuthError> {
        match self {
            Self::Static(CredentialRef::Explicit(secret)) => Ok(
                CredentialPlanResolution::Resolved(ResolvedCredential::Static(secret.clone())),
            ),
            Self::Static(CredentialRef::None) | Self::None => {
                Ok(CredentialPlanResolution::Unavailable)
            }
            Self::Static(reference) => match resolver.resolve(reference).await? {
                CredentialLookup::Found(secret) => Ok(CredentialPlanResolution::Resolved(
                    ResolvedCredential::Static(secret),
                )),
                CredentialLookup::Missing | CredentialLookup::Declined => {
                    Ok(CredentialPlanResolution::Unavailable)
                }
            },
            Self::Caller(caller) => {
                let credential = caller.acquire().await?;
                if credential.secret().expose().is_empty() {
                    return Err(AuthError::EmptyCallerCredential);
                }
                Ok(CredentialPlanResolution::Resolved(credential))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{
        CredentialLookup, CredentialLookupFuture, CredentialPlan, CredentialPlanResolution,
        CredentialRef, CredentialResolver, CredentialResolverHandle,
    };
    use crate::AuthError;
    use crate::auth::SecretValue;

    #[derive(Debug)]
    struct HostResolver;

    impl CredentialResolver for HostResolver {
        fn resolve<'a>(&'a self, reference: &'a CredentialRef) -> CredentialLookupFuture<'a> {
            Box::pin(async move {
                Ok(match reference {
                    CredentialRef::Host(name) if name == "rotating-token" => {
                        CredentialLookup::Found(SecretValue::new("resolved"))
                    }
                    _ => CredentialLookup::Declined,
                })
            })
        }
    }

    #[tokio::test]
    async fn static_host_reference_resolves_at_acquisition_time() {
        let resolver = CredentialResolverHandle::new(Arc::new(HostResolver));
        let plan = CredentialPlan::Static(CredentialRef::Host("rotating-token".to_string()));

        let resolved = plan.resolve(&resolver).await.unwrap();

        assert!(matches!(resolved, CredentialPlanResolution::Resolved(_)));
    }

    #[tokio::test]
    async fn declined_reference_is_available_for_pre_acquisition_fallback() {
        let resolver = CredentialResolverHandle::new(Arc::new(HostResolver));
        let plan = CredentialPlan::Static(CredentialRef::Request("api-key".to_string()));

        assert_eq!(
            plan.resolve(&resolver).await.unwrap(),
            CredentialPlanResolution::Unavailable
        );
    }

    #[derive(Debug)]
    struct FailingResolver;

    impl CredentialResolver for FailingResolver {
        fn resolve<'a>(&'a self, _reference: &'a CredentialRef) -> CredentialLookupFuture<'a> {
            Box::pin(async { Err(AuthError::UnresolvedOidcReference) })
        }
    }

    #[tokio::test]
    async fn acquisition_failure_is_terminal() {
        let resolver = CredentialResolverHandle::new(Arc::new(FailingResolver));
        let plan = CredentialPlan::Static(CredentialRef::Host("token".to_string()));

        let error = plan
            .resolve(&resolver)
            .await
            .expect_err("acquisition errors cannot become fallback");

        assert_eq!(error, AuthError::UnresolvedOidcReference);
    }
}
