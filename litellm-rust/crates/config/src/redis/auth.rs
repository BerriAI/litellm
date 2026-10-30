use litellm_auth_types::SecretValue;

use super::{Kwargs, Lookup, RedisError, bool_value, secret_value, text_value};

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

pub(super) fn redis_auth(merged: &Kwargs, lookup: Lookup<'_>, uses_tls: bool) -> Result<RedisAuth, RedisError> {
    if merged.contains_key("credential_provider") {
        return Ok(RedisAuth::Host);
    }
    let gcp = text_value(merged, "gcp_service_account")
        .filter(|account| !account.is_empty())
        .or_else(|| lookup("REDIS_GCP_SERVICE_ACCOUNT"));
    let azure = text_value(merged, "azure_redis_ad_token")
        .or_else(|| lookup("REDIS_AZURE_AD_TOKEN"))
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
                client_id: present(merged, "azure_client_id").or_else(|| lookup("AZURE_CLIENT_ID")),
                tenant_id: present(merged, "azure_tenant_id").or_else(|| lookup("AZURE_TENANT_ID")),
                client_secret: present(merged, "azure_client_secret")
                    .or_else(|| lookup("AZURE_CLIENT_SECRET"))
                    .map(SecretValue::new),
                username: lookup("REDIS_USERNAME").filter(|username| !username.is_empty()),
            }))
        }
        (None, false, true) => aws_iam(merged, lookup, uses_tls).map(RedisAuth::AwsIam),
        (None, false, false) => Ok(RedisAuth::Static {
            username: secret_value(merged, "username"),
            password: secret_value(merged, "password"),
        }),
    }
}

fn aws_iam(merged: &Kwargs, lookup: Lookup<'_>, uses_tls: bool) -> Result<AwsIamAuth, RedisError> {
    if !uses_tls {
        return Err(RedisError::AwsIamWithoutTls);
    }
    let user_name = present(merged, "aws_iam_user_name");
    let cache_name = present(merged, "aws_iam_cache_name");
    let region = present(merged, "aws_iam_region")
        .or_else(|| lookup("AWS_REGION").filter(|region| !region.is_empty()))
        .or_else(|| lookup("AWS_DEFAULT_REGION").filter(|region| !region.is_empty()));
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
