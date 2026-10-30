use std::time::{Duration, SystemTime};

use litellm_auth_aws::{AwsAuthConfig, AwsAuthService, SigningScope, presign_url};
use litellm_config::{AwsIamAuth, SharedLookup};

use crate::{CredentialSource, Expiring, SourceFuture};

const SERVICE: &str = "elasticache";
const TOKEN_LIFETIME: Duration = Duration::from_secs(900);

pub(crate) struct ElastiCacheIam {
    service: AwsAuthService,
    auth: AwsIamAuth,
    lookup: SharedLookup,
}

impl ElastiCacheIam {
    pub(crate) fn new(auth: AwsIamAuth, lookup: SharedLookup) -> Self {
        Self {
            service: AwsAuthService::default(),
            auth,
            lookup,
        }
    }
}

pub(crate) fn connect_url(auth: &AwsIamAuth) -> String {
    let query = [
        Some(("Action", "connect")),
        Some(("User", auth.user_name.as_str())),
        auth.serverless.then_some(("ResourceType", "ServerlessCache")),
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
            let lookup = |name: &str| (self.lookup)(name);
            let credentials = self
                .service
                .resolve_credentials(AwsAuthConfig::default(), &lookup)
                .await?;
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
            )?;
            let token_expiry = now + TOKEN_LIFETIME;
            Ok(Expiring {
                username: Some(self.auth.user_name.clone()),
                password: signed.trim_start_matches("https://").to_owned(),
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

    use super::connect_url;

    #[rstest]
    #[case::provisioned(false, "https://cache-1/?Action=connect&User=iam+user%2F1")]
    #[case::serverless(true, "https://cache-1/?Action=connect&User=iam+user%2F1&ResourceType=ServerlessCache")]
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
