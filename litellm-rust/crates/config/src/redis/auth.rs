use litellm_core_utils::settings::Lookup;
use litellm_redis_types::{AwsIamAuth, AzureAdAuth, RedisAuth};

use super::{
    RedisError,
    options::{self, AwsOptions, Options, flag, secret, text},
};

pub(super) fn redis_auth(
    options: &Options,
    environment: &dyn Lookup,
) -> Result<RedisAuth, RedisError> {
    if options.credential_provider.is_some() {
        return Ok(RedisAuth::Host);
    }
    let gcp = text(options.gcp.service_account.as_ref());
    let azure = text(options.azure.ad_token.as_ref())
        .is_some_and(|enabled| enabled.eq_ignore_ascii_case("true"));
    let aws = flag(options.aws.auth.as_ref())?;
    match (gcp, azure, aws) {
        (Some(service_account), azure, aws) => {
            if azure {
                tracing::warn!(
                    "Both GCP IAM (gcp_service_account) and Azure AD (azure_redis_ad_token) are configured for Redis. Using GCP IAM. Remove one to avoid misconfiguration."
                );
            }
            if aws {
                tracing::warn!(
                    "Both GCP IAM (gcp_service_account) and AWS ElastiCache IAM (aws_iam_auth) are configured for Redis. Using GCP IAM. Remove one to avoid misconfiguration."
                );
            }
            Ok(RedisAuth::GcpIam { service_account })
        }
        (None, true, aws) => {
            if aws {
                tracing::warn!(
                    "Both Azure AD (azure_redis_ad_token) and AWS ElastiCache IAM (aws_iam_auth) are configured for Redis. Using Azure AD. Remove one to avoid misconfiguration."
                );
            }
            Ok(RedisAuth::AzureAd(AzureAdAuth {
                client_id: text(options.azure.client_id.as_ref()),
                tenant_id: text(options.azure.tenant_id.as_ref()),
                client_secret: secret(options.azure.client_secret.as_ref()),
                username: environment
                    .get("REDIS_USERNAME")
                    .filter(|username| !username.is_empty()),
            }))
        }
        (None, false, true) => aws_iam(&options.aws, uses_tls(options)?).map(RedisAuth::AwsIam),
        (None, false, false) => Ok(RedisAuth::Static {
            username: secret(options.username.as_ref()),
            password: secret(options.password.as_ref()),
        }),
    }
}

// _uses_tls: a url's scheme decides unless cluster nodes are set, otherwise the ssl kwarg does
fn uses_tls(options: &Options) -> Result<bool, RedisError> {
    match (&options.startup_nodes, text(options.url.as_ref())) {
        (None, Some(url)) => Ok(url
            .split_once("://")
            .is_some_and(|(scheme, _)| scheme.eq_ignore_ascii_case("rediss"))),
        _ => flag(options.ssl.as_ref()),
    }
}

fn aws_iam(aws: &AwsOptions, uses_tls: bool) -> Result<AwsIamAuth, RedisError> {
    if !uses_tls {
        return Err(RedisError::AwsIamWithoutTls);
    }
    match (
        text(aws.user_name.as_ref()),
        text(aws.cache_name.as_ref()),
        text(aws.region.as_ref()),
    ) {
        (Some(user_name), Some(cache_name), Some(region)) => Ok(AwsIamAuth {
            user_name,
            cache_name,
            region,
            serverless: options::flag(aws.serverless.as_ref())?,
        }),
        (user_name, cache_name, region) => Err(RedisError::AwsIamIncomplete {
            missing: [
                ("aws_iam_user_name", user_name.is_none()),
                ("aws_iam_cache_name", cache_name.is_none()),
                ("aws_iam_region", region.is_none()),
            ]
            .into_iter()
            .filter_map(|(name, missing)| missing.then_some(name))
            .collect(),
        }),
    }
}
