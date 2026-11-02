use litellm_secret_value::SecretValue;

use super::{
    Kwargs, Lookup, RedisError, bool_value, is_set, secret_value, text_or_fallback, text_value,
};

#[derive(Clone, Debug, PartialEq)]
pub enum RedisAuth {
    Static {
        username: Option<SecretValue>,
        password: Option<SecretValue>,
    },
    Host,
    GcpIam {
        service_account: String,
    },
    AzureAd(AzureAdAuth),
    AwsIam(AwsIamAuth),
}

#[derive(Clone, Debug, PartialEq)]
pub struct AzureAdAuth {
    pub client_id: Option<String>,
    pub tenant_id: Option<String>,
    pub client_secret: Option<SecretValue>,
    pub username: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AwsIamAuth {
    pub user_name: String,
    pub cache_name: String,
    pub region: String,
    pub serverless: bool,
}

pub(super) fn redis_auth(
    merged: &Kwargs,
    lookup: &dyn Lookup,
    environment: &dyn Lookup,
    uses_tls: bool,
) -> Result<RedisAuth, RedisError> {
    if is_set(merged, "credential_provider") {
        return Ok(RedisAuth::Host);
    }
    let gcp = text_or_fallback(merged, "gcp_service_account", lookup);
    let azure = text_or_fallback(merged, "azure_redis_ad_token", lookup)
        .is_some_and(|enabled| enabled.eq_ignore_ascii_case("true"));
    let aws = bool_value(merged, "aws_iam_auth");
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
                client_id: text_or_fallback(merged, "azure_client_id", lookup),
                tenant_id: text_or_fallback(merged, "azure_tenant_id", lookup),
                client_secret: text_or_fallback(merged, "azure_client_secret", lookup)
                    .map(SecretValue::new),
                username: environment
                    .get("REDIS_USERNAME")
                    .filter(|username| !username.is_empty()),
            }))
        }
        (None, false, true) => aws_iam(merged, lookup, uses_tls).map(RedisAuth::AwsIam),
        (None, false, false) => Ok(RedisAuth::Static {
            username: secret_value(merged, "username"),
            password: secret_value(merged, "password"),
        }),
    }
}

fn aws_iam(merged: &Kwargs, lookup: &dyn Lookup, uses_tls: bool) -> Result<AwsIamAuth, RedisError> {
    if !uses_tls {
        return Err(RedisError::AwsIamWithoutTls);
    }
    let user_name = present(merged, "aws_iam_user_name");
    let cache_name = present(merged, "aws_iam_cache_name");
    let region =
        text_or_fallback(merged, "aws_iam_region", lookup).filter(|region| !region.is_empty());
    match (user_name, cache_name, region) {
        (Some(user_name), Some(cache_name), Some(region)) => Ok(AwsIamAuth {
            user_name,
            cache_name,
            region,
            serverless: bool_value(merged, "aws_iam_serverless"),
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

fn present(merged: &Kwargs, key: &str) -> Option<String> {
    text_value(merged, key).filter(|value| !value.is_empty())
}
