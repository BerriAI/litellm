use std::time::{Duration, SystemTime};

use litellm_auth_aws::{AwsAuthService, SigningScope, presign_url};
use litellm_config::AwsIamAuth;
use litellm_core_utils::settings::Lookup;

use crate::Error;
use crate::error::CredentialStage;
use litellm_secret_value::SecretValue;

use crate::{CredentialSource, Expiring, SourceFuture};

const SERVICE: &str = "elasticache";
const TOKEN_LIFETIME: Duration = Duration::from_secs(900);

const SDK_IDENTITY_ENVIRONMENT: &[&str] = &[
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_CONFIG_FILE",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_ROLE_ARN",
    "AWS_ROLE_SESSION_NAME",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
    "AWS_EC2_METADATA_DISABLED",
    "AWS_EC2_METADATA_SERVICE_ENDPOINT",
    "AWS_EC2_METADATA_SERVICE_ENDPOINT_MODE",
    "AWS_ENDPOINT_URL",
    "AWS_ENDPOINT_URL_STS",
];

pub(crate) struct ElastiCacheIam {
    service: AwsAuthService,
    auth: AwsIamAuth,
}

impl ElastiCacheIam {
    pub(crate) fn new(auth: AwsIamAuth, environment: &dyn Lookup) -> Result<Self, Error> {
        validate_environment(environment, &|name: &str| std::env::var(name).ok())?;
        Ok(Self {
            service: AwsAuthService::default(),
            auth,
        })
    }
}

fn validate_environment(environment: &dyn Lookup, process: &dyn Lookup) -> Result<(), Error> {
    match SDK_IDENTITY_ENVIRONMENT.iter().find(|name| {
        environment.get(name).is_some_and(|value| {
            Some(SecretValue::new(value)) != process.get(name).map(SecretValue::new)
        })
    }) {
        Some(setting) => Err(Error::ProcessEnvironmentRequired { setting }),
        None => Ok(()),
    }
}

pub(crate) fn connect_url(auth: &AwsIamAuth) -> String {
    let query = [
        Some(("Action", "connect")),
        Some(("User", auth.user_name.as_str())),
        auth.serverless
            .then_some(("ResourceType", "ServerlessCache")),
    ];
    let mut url = String::from("https://");
    url.push_str(&auth.cache_name.to_lowercase());
    url.push_str("/?");
    url.push_str(
        &url::form_urlencoded::Serializer::new(String::new())
            .extend_pairs(query.into_iter().flatten())
            .finish(),
    );
    url
}

impl CredentialSource for ElastiCacheIam {
    fn fetch(&self) -> SourceFuture<'_> {
        Box::pin(async move {
            let credentials = self
                .service
                .default_chain_credentials(&self.auth.region)
                .await
                .map_err(Error::credential(CredentialStage::Identity))?;
            let now = SystemTime::now();
            let signed = presign_url(
                &connect_url(&self.auth),
                TOKEN_LIFETIME,
                SigningScope {
                    region: &self.auth.region,
                    service: SERVICE,
                    credentials: &credentials,
                    time: now,
                },
            )
            .map_err(Error::credential(CredentialStage::Signing))?;
            let token_expiry = now + TOKEN_LIFETIME;
            Ok(Expiring {
                username: Some(self.auth.user_name.clone()),
                password: SecretValue::new(signed.trim_start_matches("https://")),
                expires_at: Some(
                    credentials
                        .expiry()
                        .map_or(token_expiry, |expiry| expiry.min(token_expiry)),
                ),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use litellm_config::AwsIamAuth;
    use rstest::rstest;

    use super::{connect_url, validate_environment};

    #[rstest]
    #[case::profile("AWS_PROFILE")]
    #[case::role("AWS_ROLE_ARN")]
    #[case::web_identity("AWS_WEB_IDENTITY_TOKEN_FILE")]
    #[case::source_key("AWS_ACCESS_KEY_ID")]
    #[case::source_secret("AWS_SECRET_ACCESS_KEY")]
    #[case::session("AWS_SESSION_TOKEN")]
    #[case::profile_path("AWS_SHARED_CREDENTIALS_FILE")]
    #[case::container("AWS_CONTAINER_CREDENTIALS_FULL_URI")]
    fn configured_identity_overrides_cannot_silently_use_an_ambient_identity(
        #[case] setting: &'static str,
    ) {
        let configured = |name: &str| (name == setting).then(|| "credential-value".to_owned());
        let process = |name: &str| (name == setting).then(|| "ambient".to_owned());
        let error = validate_environment(&configured, &process).unwrap_err();

        assert!(matches!(
            error,
            crate::Error::ProcessEnvironmentRequired { setting: rejected } if rejected == setting
        ));
        assert!(!error.to_string().contains("credential-value"));
        assert!(!format!("{error:?}").contains("credential-value"));
    }

    #[rstest]
    #[case::process_environment("AWS_PROFILE", Some("profile"), Some("profile"))]
    #[case::ambient_only("AWS_PROFILE", None, Some("profile"))]
    #[case::unconfigured("AWS_PROFILE", None, None)]
    #[case::unrelated_bedrock_setting("AWS_ROLE_NAME", Some("bedrock"), None)]
    fn unchanged_or_unrelated_settings_preserve_the_sdk_chain(
        #[case] setting: &str,
        #[case] configured: Option<&str>,
        #[case] ambient: Option<&str>,
    ) {
        let configured = |name: &str| {
            (name == setting)
                .then_some(configured)
                .flatten()
                .map(str::to_owned)
        };
        let ambient = |name: &str| {
            (name == setting)
                .then_some(ambient)
                .flatten()
                .map(str::to_owned)
        };
        assert!(validate_environment(&configured, &ambient).is_ok());
    }

    #[rstest]
    #[case::provisioned(false, "https://cache-1/?Action=connect&User=iam+user%2F1")]
    #[case::serverless(
        true,
        "https://cache-1/?Action=connect&User=iam+user%2F1&ResourceType=ServerlessCache"
    )]
    fn the_connect_url_matches_python_urlencode(#[case] serverless: bool, #[case] expected: &str) {
        let auth = AwsIamAuth {
            user_name: "iam user/1".to_owned(),
            cache_name: "Cache-1".to_owned(),
            region: "us-east-1".to_owned(),
            serverless,
        };

        assert_eq!(connect_url(&auth), expected);
    }
}
