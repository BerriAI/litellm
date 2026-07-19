use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::caching::in_memory_cache::InMemoryCache;
use crate::error::{CoreError, CoreResult};
use aws_credential_types::provider::ProvideCredentials;
use aws_credential_types::Credentials;
use aws_sigv4::http_request::{
    sign, SignableBody, SignableRequest, SigningParams, SigningSettings,
};
use aws_sigv4::sign::v4;
use aws_smithy_runtime_api::client::identity::Identity;
use sha2::{Digest, Sha256};

use super::constants::{
    AWS_ACCESS_KEY_ID, AWS_EXTERNAL_ID, AWS_PROFILE_NAME, AWS_REGION_NAME, AWS_ROLE_ARN,
    AWS_ROLE_NAME, AWS_SECRET_ACCESS_KEY, AWS_SESSION_NAME, AWS_SESSION_TOKEN, AWS_STS_ENDPOINT,
    AWS_WEB_IDENTITY_TOKEN, AWS_WEB_IDENTITY_TOKEN_FILE, BEDROCK_SERVICE,
    DEFAULT_SESSION_NAME_PREFIX,
};

const STATIC_CREDENTIALS_TTL: Duration = Duration::from_secs(3600 - 60);
const AMBIENT_CREDENTIALS_TTL: Duration = Duration::from_secs(600);

static IAM_CREDENTIALS_CACHE: OnceLock<Mutex<InMemoryCache<Credentials>>> = OnceLock::new();

fn credential_cache_ttl(flow: &AwsAuthFlow) -> Option<Duration> {
    match flow {
        AwsAuthFlow::StaticKeys { .. } => Some(STATIC_CREDENTIALS_TTL),
        AwsAuthFlow::DefaultChain => Some(AMBIENT_CREDENTIALS_TTL),
        AwsAuthFlow::WebIdentity { .. }
        | AwsAuthFlow::AssumeRole { .. }
        | AwsAuthFlow::Profile { .. }
        | AwsAuthFlow::SessionToken { .. } => None,
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AwsAuthConfig {
    pub access_key_id: Option<String>,
    pub secret_access_key: Option<String>,
    pub session_token: Option<String>,
    pub region_name: Option<String>,
    pub session_name: Option<String>,
    pub profile_name: Option<String>,
    pub role_name: Option<String>,
    pub web_identity_token: Option<String>,
    pub sts_endpoint: Option<String>,
    pub external_id: Option<String>,
}

impl AwsAuthConfig {
    fn with_environment(self, env_lookup: &dyn Fn(&str) -> Option<String>) -> Self {
        Self {
            access_key_id: self.access_key_id.or_else(|| env_lookup(AWS_ACCESS_KEY_ID)),
            secret_access_key: self
                .secret_access_key
                .or_else(|| env_lookup(AWS_SECRET_ACCESS_KEY)),
            session_token: self.session_token.or_else(|| env_lookup(AWS_SESSION_TOKEN)),
            region_name: self.region_name.or_else(|| env_lookup(AWS_REGION_NAME)),
            session_name: self.session_name.or_else(|| env_lookup(AWS_SESSION_NAME)),
            profile_name: self.profile_name.or_else(|| env_lookup(AWS_PROFILE_NAME)),
            role_name: self.role_name.or_else(|| env_lookup(AWS_ROLE_NAME)),
            web_identity_token: self
                .web_identity_token
                .or_else(|| env_lookup(AWS_WEB_IDENTITY_TOKEN)),
            sts_endpoint: self.sts_endpoint.or_else(|| env_lookup(AWS_STS_ENDPOINT)),
            external_id: self.external_id.or_else(|| env_lookup(AWS_EXTERNAL_ID)),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AwsAuthFlow {
    WebIdentity {
        token: String,
        role: String,
        session_name: String,
    },
    AssumeRole {
        role: String,
        session_name: Option<String>,
    },
    Profile {
        name: String,
    },
    SessionToken {
        access_key_id: String,
        secret_access_key: String,
        session_token: String,
    },
    StaticKeys {
        access_key_id: String,
        secret_access_key: String,
        region_name: String,
    },
    DefaultChain,
}

fn cache_key(config: &AwsAuthConfig, flow: &AwsAuthFlow) -> String {
    let mut hasher = Sha256::new();
    hasher.update(format!("{config:?}:{flow:?}"));
    format!("{:x}", hasher.finalize())
}

fn get_cached_credentials(key: &str) -> Option<Credentials> {
    let cache = IAM_CREDENTIALS_CACHE.get_or_init(|| Mutex::new(InMemoryCache::default()));
    let mut entries = cache.lock().ok()?;
    entries.get_cache(key)
}

fn set_cached_credentials(key: String, credentials: Credentials, ttl: Duration) {
    let cache = IAM_CREDENTIALS_CACHE.get_or_init(|| Mutex::new(InMemoryCache::default()));
    if let Ok(mut entries) = cache.lock() {
        entries.set_cache(key, credentials, Some(ttl));
    }
}

fn role_identity(arn: &str) -> Option<(&str, &str, &str)> {
    let mut parts = arn.splitn(6, ':');
    let ("arn", partition, _, _, account, resource) = (
        parts.next()?,
        parts.next()?,
        parts.next()?,
        parts.next()?,
        parts.next()?,
        parts.next()?,
    ) else {
        return None;
    };
    let role = if let Some(role) = resource.strip_prefix("role/") {
        role.rsplit('/').next()?
    } else {
        resource.strip_prefix("assumed-role/")?.split('/').next()?
    };
    Some((partition, account, role))
}

fn same_role_arns(target: &str, caller: &str) -> bool {
    role_identity(target) == role_identity(caller)
}

pub fn classify_auth(
    config: AwsAuthConfig,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> AwsAuthFlow {
    let config = config.with_environment(env_lookup);
    if let (Some(token), Some(role), Some(session_name)) = (
        config.web_identity_token.clone(),
        config.role_name.clone(),
        config.session_name.clone(),
    ) {
        return AwsAuthFlow::WebIdentity {
            token,
            role,
            session_name,
        };
    }
    if let Some(role) = config.role_name.clone() {
        return AwsAuthFlow::AssumeRole {
            role,
            session_name: config.session_name.clone(),
        };
    }
    if let Some(name) = config.profile_name {
        return AwsAuthFlow::Profile { name };
    }
    if let (Some(access_key_id), Some(secret_access_key), Some(session_token)) = (
        config.access_key_id.clone(),
        config.secret_access_key.clone(),
        config.session_token,
    ) {
        return AwsAuthFlow::SessionToken {
            access_key_id,
            secret_access_key,
            session_token,
        };
    }
    if let (Some(access_key_id), Some(secret_access_key), Some(region_name)) = (
        config.access_key_id,
        config.secret_access_key,
        config.region_name,
    ) {
        return AwsAuthFlow::StaticKeys {
            access_key_id,
            secret_access_key,
            region_name,
        };
    }
    AwsAuthFlow::DefaultChain
}

pub async fn resolve_credentials(
    config: AwsAuthConfig,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CoreResult<Credentials> {
    let resolved = config.clone().with_environment(env_lookup);
    let flow = classify_auth(config, env_lookup);
    match flow {
        AwsAuthFlow::SessionToken {
            access_key_id,
            secret_access_key,
            session_token,
        } => Ok(Credentials::new(
            access_key_id,
            secret_access_key,
            Some(session_token),
            None,
            "litellm-static-session",
        )),
        AwsAuthFlow::StaticKeys {
            access_key_id,
            secret_access_key,
            region_name,
        } => {
            let flow = AwsAuthFlow::StaticKeys {
                access_key_id: access_key_id.clone(),
                secret_access_key: secret_access_key.clone(),
                region_name,
            };
            let key = cache_key(&resolved, &flow);
            if let Some(credentials) = get_cached_credentials(&key) {
                return Ok(credentials);
            }
            let credentials = Credentials::new(
                access_key_id,
                secret_access_key,
                None,
                None,
                "litellm-static",
            );
            set_cached_credentials(
                key,
                credentials.clone(),
                credential_cache_ttl(&flow).unwrap_or(STATIC_CREDENTIALS_TTL),
            );
            Ok(credentials)
        }
        AwsAuthFlow::Profile { name } => {
            let provider = aws_config::profile::ProfileFileCredentialsProvider::builder()
                .profile_name(name)
                .build();
            provider.provide_credentials().await.map_err(|error| {
                CoreError::Auth(format!("AWS profile credentials failed: {error}"))
            })
        }
        AwsAuthFlow::AssumeRole { role, session_name } => {
            if is_already_running_as_role(&role, &resolved).await? {
                let ambient_flow = AwsAuthFlow::DefaultChain;
                let key = cache_key(&resolved, &ambient_flow);
                if let Some(credentials) = get_cached_credentials(&key) {
                    return Ok(credentials);
                }
                let provider =
                    aws_config::default_provider::credentials::DefaultCredentialsChain::builder()
                        .build()
                        .await;
                let credentials = provider.provide_credentials().await.map_err(|error| {
                    CoreError::Auth(format!("AWS default credentials failed: {error}"))
                })?;
                set_cached_credentials(
                    key,
                    credentials.clone(),
                    credential_cache_ttl(&ambient_flow).unwrap_or(AMBIENT_CREDENTIALS_TTL),
                );
                return Ok(credentials);
            }
            let mut loader = aws_config::defaults(aws_config::BehaviorVersion::latest());
            if let Some(region) = resolved.region_name.clone() {
                loader = loader.region(aws_types::region::Region::new(region));
            }
            if let Some(endpoint) = resolved.sts_endpoint.clone() {
                loader = loader.endpoint_url(endpoint);
            }
            if let (Some(access_key_id), Some(secret_access_key)) =
                (resolved.access_key_id, resolved.secret_access_key)
            {
                loader = loader.credentials_provider(Credentials::new(
                    access_key_id,
                    secret_access_key,
                    resolved.session_token,
                    None,
                    "litellm-role-source",
                ));
            }
            let sdk_config = loader.load().await;
            let builder = aws_config::sts::AssumeRoleProvider::builder(role);
            let builder = match session_name {
                Some(name) => builder.session_name(name),
                None => builder.session_name(default_session_name()),
            };
            let builder = match resolved.external_id {
                Some(id) => builder.external_id(id),
                None => builder,
            };
            let provider = builder.configure(&sdk_config).build().await;
            provider
                .provide_credentials()
                .await
                .map_err(|error| CoreError::Auth(format!("AWS role credentials failed: {error}")))
        }
        AwsAuthFlow::WebIdentity {
            token,
            role,
            session_name,
        } => {
            let mut loader = aws_config::defaults(aws_config::BehaviorVersion::latest());
            if let Some(region) = resolved.region_name {
                loader = loader.region(aws_types::region::Region::new(region));
            }
            if let Some(endpoint) = resolved.sts_endpoint {
                loader = loader.endpoint_url(endpoint);
            }
            let sdk_config = loader.load().await;
            let client = aws_sdk_sts::Client::new(&sdk_config);
            let response = client
                .assume_role_with_web_identity()
                .role_arn(role)
                .role_session_name(session_name)
                .web_identity_token(token)
                .send()
                .await
                .map_err(|error| {
                    CoreError::Auth(format!("AWS web identity credentials failed: {error}"))
                })?;
            let credentials = response.credentials().ok_or_else(|| {
                CoreError::Auth("AWS web identity response had no credentials".to_string())
            })?;
            Ok(Credentials::new(
                credentials.access_key_id(),
                credentials.secret_access_key(),
                Some(credentials.session_token().to_string()),
                None,
                "litellm-web-identity",
            ))
        }
        AwsAuthFlow::DefaultChain => {
            let key = cache_key(&resolved, &AwsAuthFlow::DefaultChain);
            if let Some(credentials) = get_cached_credentials(&key) {
                return Ok(credentials);
            }
            let provider =
                aws_config::default_provider::credentials::DefaultCredentialsChain::builder()
                    .build()
                    .await;
            let credentials = provider.provide_credentials().await.map_err(|error| {
                CoreError::Auth(format!("AWS default credentials failed: {error}"))
            })?;
            set_cached_credentials(
                key,
                credentials.clone(),
                credential_cache_ttl(&AwsAuthFlow::DefaultChain).unwrap_or(AMBIENT_CREDENTIALS_TTL),
            );
            Ok(credentials)
        }
    }
}

async fn is_already_running_as_role(role: &str, config: &AwsAuthConfig) -> CoreResult<bool> {
    if role_identity(role).is_none() {
        return Ok(false);
    }
    if let (Ok(current_role), Ok(token_file)) = (
        std::env::var(AWS_ROLE_ARN),
        std::env::var(AWS_WEB_IDENTITY_TOKEN_FILE),
    ) {
        if !token_file.is_empty() {
            return Ok(same_role_arns(role, &current_role));
        }
    }
    let mut loader = aws_config::defaults(aws_config::BehaviorVersion::latest());
    if let Some(region) = config.region_name.clone() {
        loader = loader.region(aws_types::region::Region::new(region));
    }
    if let Some(endpoint) = config.sts_endpoint.clone() {
        loader = loader.endpoint_url(endpoint);
    }
    let sdk_config = loader.load().await;
    let response = match aws_sdk_sts::Client::new(&sdk_config)
        .get_caller_identity()
        .send()
        .await
    {
        Ok(response) => response,
        Err(_) => return Ok(false),
    };
    Ok(response
        .arn()
        .is_some_and(|caller| same_role_arns(role, caller)))
}

fn default_session_name() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs());
    format!("{DEFAULT_SESSION_NAME_PREFIX}-{seconds}")
}

pub fn sign_bedrock_post(
    url: &str,
    body: &[u8],
    headers: &BTreeMap<String, String>,
    region: &str,
    credentials: &Credentials,
    signing_time: SystemTime,
) -> CoreResult<BTreeMap<String, String>> {
    let identity: Identity = credentials.clone().into();
    let params = v4::SigningParams::builder()
        .identity(&identity)
        .region(region)
        .name(BEDROCK_SERVICE)
        .time(signing_time)
        .settings(SigningSettings::default())
        .build()
        .map(SigningParams::from)
        .map_err(|error| CoreError::Auth(format!("AWS signing parameters failed: {error}")))?;
    let header_refs = headers
        .iter()
        .map(|(name, value)| (name.as_str(), value.as_str()));
    let request = SignableRequest::new("POST", url, header_refs, SignableBody::Bytes(body))
        .map_err(|error| CoreError::Auth(format!("AWS signable request failed: {error}")))?;
    let (instructions, _) = sign(request, &params)
        .map_err(|error| CoreError::Auth(format!("AWS request signing failed: {error}")))?
        .into_parts();
    Ok(instructions
        .headers()
        .map(|(name, value)| {
            let normalized_name = match name {
                "authorization" => "Authorization",
                "x-amz-date" => "X-Amz-Date",
                "x-amz-security-token" => "X-Amz-Security-Token",
                _ => name,
            };
            (normalized_name.to_string(), value.to_string())
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn parity_inputs() -> (String, Vec<u8>, BTreeMap<String, String>) {
        (
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.titan-text-express-v1/invoke"
                .to_string(),
            br#"{"input":"hello"}"#.to_vec(),
            BTreeMap::from([("Content-Type".to_string(), "application/json".to_string())]),
        )
    }

    #[test]
    fn classification_preserves_python_precedence() {
        let config = AwsAuthConfig {
            access_key_id: Some("ak".into()),
            secret_access_key: Some("sk".into()),
            session_token: Some("token".into()),
            region_name: Some("us-east-1".into()),
            session_name: Some("session".into()),
            profile_name: Some("profile".into()),
            role_name: Some("role".into()),
            web_identity_token: Some("oidc".into()),
            ..Default::default()
        };
        assert!(matches!(
            classify_auth(config, &no_env),
            AwsAuthFlow::WebIdentity { .. }
        ));
    }

    #[test]
    fn classification_covers_fallthroughs() {
        let env = |key: &str| match key {
            AWS_PROFILE_NAME => Some("profile".into()),
            _ => None,
        };
        assert!(matches!(
            classify_auth(AwsAuthConfig::default(), &env),
            AwsAuthFlow::Profile { .. }
        ));
        assert!(matches!(
            classify_auth(
                AwsAuthConfig {
                    access_key_id: Some("ak".into()),
                    secret_access_key: Some("sk".into()),
                    session_token: Some("token".into()),
                    ..Default::default()
                },
                &no_env
            ),
            AwsAuthFlow::SessionToken { .. }
        ));
        assert!(matches!(
            classify_auth(
                AwsAuthConfig {
                    access_key_id: Some("ak".into()),
                    secret_access_key: Some("sk".into()),
                    region_name: Some("us-east-1".into()),
                    ..Default::default()
                },
                &no_env
            ),
            AwsAuthFlow::StaticKeys { .. }
        ));
        assert_eq!(
            classify_auth(AwsAuthConfig::default(), &no_env),
            AwsAuthFlow::DefaultChain
        );
    }

    #[tokio::test]
    async fn static_credentials_do_not_use_network() {
        let credentials = resolve_credentials(
            AwsAuthConfig {
                access_key_id: Some("ak".into()),
                secret_access_key: Some("sk".into()),
                region_name: Some("us-east-1".into()),
                ..Default::default()
            },
            &no_env,
        )
        .await
        .expect("static credentials");
        assert_eq!(credentials.access_key_id(), "ak");
        assert_eq!(credentials.session_token(), None);
    }

    #[test]
    fn cache_policy_matches_python_flows() {
        assert_eq!(
            credential_cache_ttl(&AwsAuthFlow::StaticKeys {
                access_key_id: "ak".into(),
                secret_access_key: "sk".into(),
                region_name: "us-east-1".into(),
            }),
            Some(STATIC_CREDENTIALS_TTL)
        );
        assert_eq!(
            credential_cache_ttl(&AwsAuthFlow::DefaultChain),
            Some(AMBIENT_CREDENTIALS_TTL)
        );
        assert_eq!(
            credential_cache_ttl(&AwsAuthFlow::SessionToken {
                access_key_id: "ak".into(),
                secret_access_key: "sk".into(),
                session_token: "token".into(),
            }),
            None
        );
        assert_eq!(
            credential_cache_ttl(&AwsAuthFlow::Profile {
                name: "profile".into()
            }),
            None
        );
        assert_eq!(
            credential_cache_ttl(&AwsAuthFlow::AssumeRole {
                role: "arn:aws:iam::123456789012:role/demo".into(),
                session_name: None,
            }),
            None
        );
        assert_eq!(
            credential_cache_ttl(&AwsAuthFlow::WebIdentity {
                token: "token".into(),
                role: "arn:aws:iam::123456789012:role/demo".into(),
                session_name: "session".into(),
            }),
            None
        );
    }

    #[test]
    fn cache_round_trip_preserves_credentials() {
        let key = format!("cache-test-{}", std::process::id());
        let credentials = Credentials::new("cache-ak", "cache-sk", None, None, "test");
        set_cached_credentials(key.clone(), credentials.clone(), STATIC_CREDENTIALS_TTL);
        assert_eq!(
            get_cached_credentials(&key).map(|value| value.access_key_id().to_string()),
            Some("cache-ak".to_string())
        );
    }

    #[test]
    fn same_role_comparison_matches_partition_account_and_role() {
        assert!(same_role_arns(
            "arn:aws:iam::123456789012:role/path/demo",
            "arn:aws:sts::123456789012:assumed-role/demo/session"
        ));
        assert!(!same_role_arns(
            "arn:aws:iam::123456789012:role/demo",
            "arn:aws:iam::999999999999:role/demo"
        ));
        assert!(!same_role_arns(
            "arn:aws:iam::123456789012:role/demo",
            "arn:aws-cn:iam::123456789012:role/demo"
        ));
        assert!(!same_role_arns(
            "arn:aws:iam::123456789012:user/demo",
            "arn:aws:iam::123456789012:role/demo"
        ));
    }

    #[test]
    fn signing_matches_botocore_golden_vector() {
        let (url, body, headers) = parity_inputs();
        let credentials = Credentials::new(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            Some("session-token".to_string()),
            None,
            "test",
        );
        let signed = sign_bedrock_post(
            &url,
            &body,
            &headers,
            "us-east-1",
            &credentials,
            UNIX_EPOCH + std::time::Duration::from_secs(1_704_164_645),
        )
        .expect("golden signature");
        assert_eq!(
            signed.get("X-Amz-Date").map(String::as_str),
            Some("20240102T030405Z")
        );
        assert_eq!(
            signed.get("X-Amz-Security-Token").map(String::as_str),
            Some("session-token")
        );
        assert_eq!(
            signed.get("Authorization").map(String::as_str),
            Some("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20240102/us-east-1/bedrock/aws4_request, SignedHeaders=content-type;host;x-amz-date;x-amz-security-token, Signature=55c027ef47527d3ad63f1735f9d099efdbc99f296ff914bd94e727e24ec0e464")
        );
    }

    #[test]
    fn signing_without_session_token_omits_security_header() {
        let (url, body, headers) = parity_inputs();
        let credentials = Credentials::new(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            None,
            None,
            "test",
        );
        let signed = sign_bedrock_post(
            &url,
            &body,
            &headers,
            "us-east-1",
            &credentials,
            UNIX_EPOCH + std::time::Duration::from_secs(1_704_164_645),
        )
        .expect("signature");
        assert!(!signed.contains_key("X-Amz-Security-Token"));
    }

    #[ignore]
    #[tokio::test]
    async fn live_bedrock_invoke_model_returns_200() -> Result<(), Box<dyn std::error::Error>> {
        let access_key_id = std::env::var("AWS_BEDROCK_TEST_ACCESS_KEY_ID")?;
        let secret_access_key = std::env::var("AWS_BEDROCK_TEST_SECRET_ACCESS_KEY")?;
        let body = br#"{"anthropic_version":"bedrock-2023-05-31","max_tokens":1,"messages":[{"role":"user","content":[{"type":"text","text":"ping"}]}]}"#.to_vec();
        let headers =
            BTreeMap::from([("Content-Type".to_string(), "application/json".to_string())]);
        let credentials = resolve_credentials(
            AwsAuthConfig {
                access_key_id: Some(access_key_id),
                secret_access_key: Some(secret_access_key),
                region_name: Some("us-west-2".to_string()),
                ..Default::default()
            },
            &no_env,
        )
        .await?;
        let client = reqwest::Client::new();
        let mut failures = Vec::new();

        for region in ["us-west-2", "us-east-1"] {
            let url = format!(
                "https://bedrock-runtime.{region}.amazonaws.com/model/us.anthropic.claude-opus-4-8/invoke"
            );
            let signed_headers = sign_bedrock_post(
                &url,
                &body,
                &headers,
                region,
                &credentials,
                SystemTime::now(),
            )?;
            let mut request = client.post(&url).body(body.clone());
            for (name, value) in &headers {
                request = request.header(name, value);
            }
            for (name, value) in signed_headers {
                request = request.header(name, value);
            }
            let response = request.send().await?;
            let status = response.status();
            let response_body = response.text().await?;
            let snippet: String = response_body.chars().take(240).collect();
            println!("region={region} status={status} response={snippet}");
            if status == reqwest::StatusCode::OK {
                return Ok(());
            }
            failures.push(format!("{region}: {status} {snippet}"));
        }

        panic!(
            "no Bedrock region returned HTTP 200: {}",
            failures.join("; ")
        );
    }
}
