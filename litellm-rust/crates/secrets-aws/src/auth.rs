use std::sync::Arc;

use aws_credential_types::provider::{ProvideCredentials, error::CredentialsError, future};
use litellm_auth_aws::{
    AwsAuthConfig, AwsAuthService,
    constants::{AWS_DEFAULT_REGION, AWS_REGION, AWS_REGION_NAME},
};
use litellm_core_utils::settings::Lookup;
use litellm_secrets_types::{AwsOperationContext, KeyManagementSettings};

use crate::Error;

#[derive(Clone)]
pub(crate) struct Credentials {
    auth: AwsAuthService,
    config: AwsAuthConfig,
    environment: Arc<dyn Lookup + Send + Sync>,
}

impl Credentials {
    pub(crate) fn new(
        settings: &KeyManagementSettings,
        environment: Arc<dyn Lookup + Send + Sync>,
    ) -> Self {
        Self::with_context(
            AwsAuthService::default(),
            settings,
            environment,
            &AwsOperationContext::default(),
        )
    }

    pub(crate) fn with_context(
        auth: AwsAuthService,
        settings: &KeyManagementSettings,
        environment: Arc<dyn Lookup + Send + Sync>,
        context: &AwsOperationContext,
    ) -> Self {
        Self {
            auth,
            config: AwsAuthConfig {
                access_key_id: context
                    .access_key_id
                    .as_ref()
                    .map(|value| value.expose().to_owned()),
                secret_access_key: context
                    .secret_access_key
                    .as_ref()
                    .map(|value| value.expose().to_owned()),
                session_token: context
                    .session_token
                    .as_ref()
                    .map(|value| value.expose().to_owned()),
                region_name: region(settings, environment.as_ref()).ok(),
                role_name: settings.aws_role_name.clone(),
                session_name: settings.aws_session_name.clone(),
                external_id: settings
                    .aws_external_id
                    .as_ref()
                    .map(|v| v.expose().to_owned()),
                profile_name: settings.aws_profile_name.clone(),
                web_identity_token: settings
                    .aws_web_identity_token
                    .as_ref()
                    .map(|v| v.expose().to_owned()),
                sts_endpoint: settings.aws_sts_endpoint.clone(),
            },
            environment,
        }
    }
}

impl ProvideCredentials for Credentials {
    fn provide_credentials<'a>(&'a self) -> future::ProvideCredentials<'a>
    where
        Self: 'a,
    {
        future::ProvideCredentials::new(async {
            self.auth
                .resolve_credentials(self.config.clone(), &|name| self.environment.get(name))
                .await
                .map_err(|_| {
                    CredentialsError::provider_error("secret manager authentication failed")
                })
        })
    }
}

pub(crate) fn region(
    settings: &KeyManagementSettings,
    environment: &dyn Lookup,
) -> Result<String, Error> {
    settings
        .aws_region_name
        .clone()
        .or_else(|| environment.get(AWS_REGION_NAME))
        .or_else(|| environment.get(AWS_REGION))
        .or_else(|| environment.get(AWS_DEFAULT_REGION))
        .ok_or(Error::MissingRegion)
}

impl std::fmt::Debug for Credentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Credentials").finish_non_exhaustive()
    }
}
