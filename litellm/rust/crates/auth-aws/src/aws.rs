use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use aws_credential_types::provider::{SharedCredentialsProvider, error::CredentialsError};
use aws_smithy_runtime_api::client::http::SharedHttpClient;
use litellm_secret_value::SecretValue;
use moka::{Expiry, future::Cache};
use std::{future::Future, pin::Pin, sync::Arc};

use aws_credential_types::Credentials;
use aws_credential_types::provider::ProvideCredentials;
use aws_sigv4::http_request::{
    SignableBody, SignableRequest, SignatureLocation, SigningInstructions, SigningParams,
    SigningSettings, sign,
};
use aws_sigv4::sign::v4;
use aws_smithy_runtime_api::client::identity::Identity;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};

use litellm_auth_types::{AwsParams, AwsSessionTag, ParamSpec};

use super::constants::{
    AWS_ACCESS_KEY_ID, AWS_DEFAULT_REGION, AWS_EXTERNAL_ID, AWS_PROFILE_NAME, AWS_REGION,
    AWS_REGION_NAME, AWS_ROLE_ARN, AWS_ROLE_NAME, AWS_SECRET_ACCESS_KEY, AWS_SESSION_NAME,
    AWS_SESSION_TOKEN, AWS_SIGNED_HEADER_NAMES, AWS_STS_ENDPOINT, AWS_WEB_IDENTITY_TOKEN,
    AWS_WEB_IDENTITY_TOKEN_FILE, BEDROCK_SESSION_POLICY, DEFAULT_BEDROCK_REGION,
    DEFAULT_SESSION_NAME_PREFIX, SIGV4_COMPUTED_HEADER_NAMES, WEB_IDENTITY_SESSION_SECONDS,
};
use crate::error::{CredentialFlow, Error, SigningStage};
use crate::region::{session_region, sts_endpoint_region};

const CREDENTIAL_CACHE_TTL: Duration = Duration::from_secs(600);
const EXPIRY_BUFFER: Duration = Duration::from_secs(60);

pub type WebIdentityTokenFuture<'a> =
    Pin<Box<dyn Future<Output = Result<Option<SecretValue>, litellm_error::Source>> + Send + 'a>>;

pub trait WebIdentityTokenSource: Send + Sync + std::fmt::Debug {
    fn web_identity_token<'a>(&'a self, reference: &'a str) -> WebIdentityTokenFuture<'a>;
}

#[derive(Clone)]
pub struct AwsAuthService {
    providers: Cache<AwsAuthConfig, SharedCredentialsProvider>,
    credentials: Cache<AwsAuthConfig, Credentials>,
    loader: Arc<dyn ProviderLoader>,
}

impl Default for AwsAuthService {
    fn default() -> Self {
        Self::new(Arc::new(NativeProviderLoader::default()))
    }
}

impl AwsAuthService {
    fn new(loader: Arc<dyn ProviderLoader>) -> Self {
        Self {
            providers: Cache::builder().max_capacity(200).build(),
            credentials: Cache::builder()
                .max_capacity(200)
                .expire_after(CredentialExpiry)
                .build(),
            loader,
        }
    }

    pub fn with_web_identity_tokens(tokens: Arc<dyn WebIdentityTokenSource>) -> Self {
        Self::new(Arc::new(NativeProviderLoader {
            tokens: Some(tokens),
            ..NativeProviderLoader::default()
        }))
    }
}

struct CredentialExpiry;

impl Expiry<AwsAuthConfig, Credentials> for CredentialExpiry {
    fn expire_after_create(
        &self,
        _: &AwsAuthConfig,
        credentials: &Credentials,
        _: std::time::Instant,
    ) -> Option<Duration> {
        Some(credentials.expiry().map_or(CREDENTIAL_CACHE_TTL, |expiry| {
            expiry
                .duration_since(SystemTime::now())
                .unwrap_or_default()
                .saturating_sub(EXPIRY_BUFFER)
                .min(CREDENTIAL_CACHE_TTL)
        }))
    }
}

type ProviderFuture<'a> =
    Pin<Box<dyn Future<Output = Result<SharedCredentialsProvider, Error>> + Send + 'a>>;

trait ProviderLoader: Send + Sync {
    fn load(&self, config: AwsAuthConfig, flow: AwsAuthFlow) -> ProviderFuture<'_>;
}

#[derive(Clone, Default)]
struct NativeProviderLoader {
    tokens: Option<Arc<dyn WebIdentityTokenSource>>,
    http_client: Option<SharedHttpClient>,
    ambient: Option<SharedCredentialsProvider>,
}

impl std::fmt::Debug for NativeProviderLoader {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeProviderLoader")
            .finish_non_exhaustive()
    }
}

impl ProviderLoader for NativeProviderLoader {
    fn load(&self, config: AwsAuthConfig, flow: AwsAuthFlow) -> ProviderFuture<'_> {
        Box::pin(build_provider(config, flow, self))
    }
}

impl NativeProviderLoader {
    async fn ambient(&self) -> SharedCredentialsProvider {
        match &self.ambient {
            Some(provider) => provider.clone(),
            None => SharedCredentialsProvider::new(
                aws_config::default_provider::credentials::DefaultCredentialsChain::builder()
                    .build()
                    .await,
            ),
        }
    }

    async fn sts_config(
        &self,
        config: &AwsAuthConfig,
        region: Option<String>,
    ) -> aws_types::SdkConfig {
        let loader = aws_config::defaults(aws_config::BehaviorVersion::latest());
        let loader = match region {
            Some(region) => loader.region(aws_types::region::Region::new(region)),
            None => loader,
        };
        let loader = match config.sts_endpoint.clone() {
            Some(endpoint) => loader.endpoint_url(endpoint),
            None => loader,
        };
        let loader = match self.http_client.clone() {
            Some(client) => loader.http_client(client),
            None => loader,
        };
        let loader = match (&config.access_key_id, &config.secret_access_key) {
            (Some(key), Some(secret)) => loader.credentials_provider(Credentials::new(
                key.expose(),
                secret.expose(),
                config
                    .session_token
                    .as_ref()
                    .map(|token| token.expose().to_owned()),
                None,
                "litellm-role-source",
            )),
            _ => match &self.ambient {
                Some(ambient) => loader.credentials_provider(ambient.clone()),
                None => loader,
            },
        };
        loader.load().await
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct AwsAuthConfig {
    pub access_key_id: Option<SecretValue>,
    pub secret_access_key: Option<SecretValue>,
    pub session_token: Option<SecretValue>,
    pub region_name: Option<String>,
    pub session_name: Option<String>,
    pub profile_name: Option<String>,
    pub role_name: Option<String>,
    pub web_identity_token: Option<SecretValue>,
    pub sts_endpoint: Option<String>,
    pub external_id: Option<SecretValue>,
    pub session_tags: Option<Vec<AwsSessionTag>>,
    pub session_policy: Option<&'static str>,
}

impl AwsAuthConfig {
    fn with_environment(self, env_lookup: &(dyn Fn(&str) -> Option<String> + Sync)) -> Self {
        Self {
            access_key_id: self
                .access_key_id
                .or_else(|| env_lookup(AWS_ACCESS_KEY_ID).map(SecretValue::new)),
            secret_access_key: self
                .secret_access_key
                .or_else(|| env_lookup(AWS_SECRET_ACCESS_KEY).map(SecretValue::new)),
            session_token: self
                .session_token
                .or_else(|| env_lookup(AWS_SESSION_TOKEN).map(SecretValue::new)),
            region_name: self.region_name.or_else(|| env_lookup(AWS_REGION_NAME)),
            session_name: self.session_name.or_else(|| env_lookup(AWS_SESSION_NAME)),
            profile_name: self.profile_name.or_else(|| env_lookup(AWS_PROFILE_NAME)),
            role_name: self.role_name.or_else(|| env_lookup(AWS_ROLE_NAME)),
            web_identity_token: self
                .web_identity_token
                .or_else(|| env_lookup(AWS_WEB_IDENTITY_TOKEN).map(SecretValue::new)),
            sts_endpoint: self.sts_endpoint.or_else(|| env_lookup(AWS_STS_ENDPOINT)),
            external_id: self
                .external_id
                .or_else(|| env_lookup(AWS_EXTERNAL_ID).map(SecretValue::new)),
            session_tags: self.session_tags,
            session_policy: self.session_policy,
        }
    }

    fn sts_region(&self, env_lookup: &(dyn Fn(&str) -> Option<String> + Sync)) -> Option<String> {
        let present = |value: Option<String>| value.filter(|value| !value.is_empty());
        self.sts_endpoint
            .as_deref()
            .and_then(sts_endpoint_region)
            .or_else(|| present(env_lookup(AWS_REGION)))
            .or_else(|| present(env_lookup(AWS_DEFAULT_REGION)))
            .or_else(|| present(self.region_name.clone()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AwsAuthFlow {
    WebIdentity {
        token: SecretValue,
        role: String,
        session_name: String,
        sts_region: Option<String>,
    },
    AssumeRole {
        role: String,
        session_name: Option<String>,
        web_identity_role: Option<String>,
        sts_region: Option<String>,
    },
    Profile {
        name: String,
    },
    SessionToken {
        access_key_id: SecretValue,
        secret_access_key: SecretValue,
        session_token: SecretValue,
    },
    StaticKeys {
        access_key_id: SecretValue,
        secret_access_key: SecretValue,
        region_name: String,
    },
    DefaultChain,
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
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> AwsAuthFlow {
    let web_identity_role = env_lookup(AWS_WEB_IDENTITY_TOKEN_FILE)
        .filter(|token_file| !token_file.is_empty())
        .and_then(|_| env_lookup(AWS_ROLE_ARN));
    let config = config.with_environment(env_lookup);
    let sts_region = config.sts_region(env_lookup);
    if let (Some(token), Some(role), Some(session_name)) = (
        config.web_identity_token.clone(),
        config.role_name.clone(),
        config.session_name.clone(),
    ) {
        return AwsAuthFlow::WebIdentity {
            token,
            role,
            session_name,
            sts_region,
        };
    }
    if let Some(role) = config.role_name.clone() {
        return AwsAuthFlow::AssumeRole {
            role,
            session_name: config.session_name.clone(),
            web_identity_role,
            sts_region,
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

async fn profile_provider(
    name: &str,
    region: Option<String>,
    config: aws_config::provider_config::ProviderConfig,
    builder: aws_config::profile::credentials::Builder,
) -> Result<SharedCredentialsProvider, Error> {
    let region = match region {
        Some(region) => Some(aws_types::region::Region::new(region)),
        None => {
            aws_config::default_provider::region::DefaultRegionChain::builder()
                .profile_name(name)
                .build()
                .region()
                .await
        }
    };
    Ok(SharedCredentialsProvider::new(
        builder
            .configure(&config.with_region(region))
            .profile_name(name)
            .build(),
    ))
}

impl AwsAuthService {
    pub async fn resolve_credentials(
        &self,
        config: AwsAuthConfig,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Credentials, Error> {
        let resolved = config.clone().with_environment(env_lookup);
        let flow = classify_auth(config, env_lookup);
        if let AwsAuthFlow::WebIdentity { token, .. } = &flow
            && names_the_environment(token.expose(), env_lookup)
        {
            return Err(Error::WebIdentityTokenReference);
        }
        self.fresh_credentials(&resolved, &flow).await
    }

    /// The credentials botocore's default chain finds in `region`, without LiteLLM's Bedrock settings
    /// (`AWS_ROLE_NAME`, `AWS_PROFILE_NAME`, `AWS_SESSION_NAME`) that `resolve_credentials` also reads.
    pub async fn default_chain_credentials(&self, region: &str) -> Result<Credentials, Error> {
        let config = AwsAuthConfig {
            region_name: Some(region.to_owned()),
            ..Default::default()
        };
        self.fresh_credentials(&config, &AwsAuthFlow::DefaultChain)
            .await
    }

    async fn fresh_credentials(
        &self,
        resolved: &AwsAuthConfig,
        flow: &AwsAuthFlow,
    ) -> Result<Credentials, Error> {
        let cached = self.cached_credentials(resolved, flow).await?;
        if fresh(&cached) {
            return Ok(cached);
        }
        self.credentials.invalidate(resolved).await;
        let refetched = self.cached_credentials(resolved, flow).await?;
        if fresh(&refetched) {
            return Ok(refetched);
        }
        self.credentials.invalidate(resolved).await;
        Err(Error::ExpiredCredentials)
    }

    async fn cached_credentials(
        &self,
        resolved: &AwsAuthConfig,
        flow: &AwsAuthFlow,
    ) -> Result<Credentials, Error> {
        self.credentials
            .try_get_with(resolved.clone(), async {
                let provider = self
                    .providers
                    .try_get_with(
                        resolved.clone(),
                        self.loader.load(resolved.clone(), flow.clone()),
                    )
                    .await
                    .map_err(|error| (*error).clone())?;
                provider
                    .provide_credentials()
                    .await
                    .map_err(Error::credentials(flow.credential_flow()))
            })
            .await
            .map_err(|error| (*error).clone())
    }
}

fn names_the_environment(
    reference: &str,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> bool {
    reference.starts_with("os.environ/") || env_lookup(reference).is_some()
}

fn fresh(credentials: &Credentials) -> bool {
    credentials
        .expiry()
        .is_none_or(|expiry| expiry > SystemTime::now() + EXPIRY_BUFFER)
}

impl AwsAuthFlow {
    fn credential_flow(&self) -> CredentialFlow {
        match self {
            Self::Profile { .. } => CredentialFlow::Profile,
            Self::AssumeRole { .. } => CredentialFlow::AssumeRole,
            Self::WebIdentity { .. } => CredentialFlow::WebIdentity,
            Self::DefaultChain | Self::StaticKeys { .. } | Self::SessionToken { .. } => {
                CredentialFlow::DefaultChain
            }
        }
    }
}

async fn build_provider(
    resolved: AwsAuthConfig,
    flow: AwsAuthFlow,
    loader: &NativeProviderLoader,
) -> Result<SharedCredentialsProvider, Error> {
    match flow {
        AwsAuthFlow::SessionToken {
            access_key_id,
            secret_access_key,
            session_token,
        } => Ok(SharedCredentialsProvider::new(Credentials::new(
            access_key_id.expose(),
            secret_access_key.expose(),
            Some(session_token.expose().to_owned()),
            None,
            "litellm-static-session",
        ))),
        AwsAuthFlow::StaticKeys {
            access_key_id,
            secret_access_key,
            ..
        } => Ok(SharedCredentialsProvider::new(Credentials::new(
            access_key_id.expose(),
            secret_access_key.expose(),
            None,
            None,
            "litellm-static",
        ))),
        AwsAuthFlow::Profile { name } => {
            profile_provider(
                &name,
                resolved.region_name,
                aws_config::provider_config::ProviderConfig::without_region(),
                aws_config::profile::ProfileFileCredentialsProvider::builder(),
            )
            .await
        }
        AwsAuthFlow::DefaultChain => {
            let builder =
                aws_config::default_provider::credentials::DefaultCredentialsChain::builder();
            let builder = match resolved.region_name {
                Some(region) => builder.region(aws_types::region::Region::new(region)),
                None => builder,
            };
            Ok(SharedCredentialsProvider::new(builder.build().await))
        }
        AwsAuthFlow::AssumeRole {
            role,
            session_name,
            web_identity_role,
            sts_region,
        } => {
            let running_as = RunningAs {
                role: role.clone(),
                config: resolved.clone(),
                sts_region: sts_region.clone(),
                loader: loader.clone(),
            };
            if running_as.is_target(web_identity_role.as_deref()).await {
                return Ok(loader.ambient().await);
            }
            let sdk_config = loader.sts_config(&resolved, sts_region).await;
            let builder = aws_config::sts::AssumeRoleProvider::builder(role)
                .session_name(session_name.unwrap_or_else(default_session_name));
            let builder = match resolved.external_id {
                Some(id) => builder.external_id(id.expose()),
                None => builder,
            };
            let builder = match resolved.session_tags {
                Some(tags) => builder.tags(tags.into_iter().map(|tag| (tag.key, tag.value))),
                None => builder,
            };
            Ok(SharedCredentialsProvider::new(AssumedRole {
                provider: builder.configure(&sdk_config).build().await,
                fallback: web_identity_role.is_none().then_some(running_as),
            }))
        }
        AwsAuthFlow::WebIdentity {
            token,
            role,
            session_name,
            sts_region,
        } => Ok(SharedCredentialsProvider::new(WebIdentityProvider {
            client: aws_sdk_sts::Client::new(&loader.sts_config(&resolved, sts_region).await),
            tokens: loader.tokens.clone(),
            token,
            role,
            session_name,
            policy: resolved.session_policy,
        })),
    }
}

#[derive(Debug)]
struct RunningAs {
    role: String,
    config: AwsAuthConfig,
    sts_region: Option<String>,
    loader: NativeProviderLoader,
}

impl RunningAs {
    async fn is_target(&self, web_identity_role: Option<&str>) -> bool {
        if self.config.access_key_id.is_some() || self.config.secret_access_key.is_some() {
            return false;
        }
        if role_identity(&self.role).is_none() {
            return false;
        }
        if let Some(current_role) = web_identity_role {
            return same_role_arns(&self.role, current_role);
        }
        let caller = AwsAuthConfig {
            sts_endpoint: self.config.sts_endpoint.clone(),
            ..Default::default()
        };
        let sdk_config = self
            .loader
            .sts_config(&caller, self.sts_region.clone())
            .await;
        aws_sdk_sts::Client::new(&sdk_config)
            .get_caller_identity()
            .send()
            .await
            .ok()
            .and_then(|response| response.arn)
            .is_some_and(|caller| same_role_arns(&self.role, &caller))
    }
}

#[derive(Debug)]
struct AssumedRole {
    provider: aws_config::sts::AssumeRoleProvider,
    fallback: Option<RunningAs>,
}

impl ProvideCredentials for AssumedRole {
    fn provide_credentials<'a>(
        &'a self,
    ) -> aws_credential_types::provider::future::ProvideCredentials<'a>
    where
        Self: 'a,
    {
        aws_credential_types::provider::future::ProvideCredentials::new(async {
            let error = match self.provider.provide_credentials().await {
                Ok(credentials) => return Ok(credentials),
                Err(error) => error,
            };
            match &self.fallback {
                Some(running_as)
                    if format!("{error:?}").contains("AccessDenied")
                        && running_as.is_target(None).await =>
                {
                    running_as
                        .loader
                        .ambient()
                        .await
                        .provide_credentials()
                        .await
                }
                _ => Err(error),
            }
        })
    }
}

#[derive(Debug)]
struct WebIdentityProvider {
    client: aws_sdk_sts::Client,
    tokens: Option<Arc<dyn WebIdentityTokenSource>>,
    token: SecretValue,
    role: String,
    session_name: String,
    policy: Option<&'static str>,
}

impl WebIdentityProvider {
    async fn token(&self) -> Result<SecretValue, Error> {
        let Some(tokens) = &self.tokens else {
            return Ok(self.token.clone());
        };
        tokens
            .web_identity_token(self.token.expose())
            .await
            .map_err(Error::WebIdentityToken)?
            .ok_or(Error::MissingWebIdentityToken)
    }
}

impl ProvideCredentials for WebIdentityProvider {
    fn provide_credentials<'a>(
        &'a self,
    ) -> aws_credential_types::provider::future::ProvideCredentials<'a>
    where
        Self: 'a,
    {
        aws_credential_types::provider::future::ProvideCredentials::new(async {
            let token = self
                .token()
                .await
                .map_err(CredentialsError::provider_error)?;
            let response = self
                .client
                .assume_role_with_web_identity()
                .role_arn(&self.role)
                .role_session_name(&self.session_name)
                .web_identity_token(token.expose())
                .duration_seconds(WEB_IDENTITY_SESSION_SECONDS)
                .set_policy(self.policy.map(str::to_owned))
                .send()
                .await
                .map_err(CredentialsError::provider_error)?;
            let credentials = response.credentials().ok_or_else(|| {
                CredentialsError::provider_error(Error::MissingWebIdentityCredentials)
            })?;
            let expiration = SystemTime::try_from(*credentials.expiration())
                .map_err(CredentialsError::provider_error)?;
            Ok(Credentials::new(
                credentials.access_key_id(),
                credentials.secret_access_key(),
                Some(credentials.session_token().to_owned()),
                Some(expiration),
                "litellm-web-identity",
            ))
        })
    }
}

fn default_session_name() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs());
    format!("{DEFAULT_SESSION_NAME_PREFIX}-{seconds}")
}

/// The subset of `headers` SigV4 should cover.
///
/// Python signs only these and reattaches the rest afterwards, so a forwarded
/// client header cannot change the canonical request and invalidate the
/// signature. Signing everything instead makes the request 403 on a header the
/// caller supplied, on a deployment that works on the Python path.
pub fn aws_signature_headers(headers: &HeaderMap) -> HeaderMap {
    headers
        .iter()
        .filter(|(name, _)| {
            let name = name.as_str();
            AWS_SIGNED_HEADER_NAMES.contains(&name)
                || name.starts_with("x-amz-")
                || name.starts_with("x-amzn-")
        })
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

/// Whether the signer produces `name` itself.
///
/// Python's reattach loop skips these, so a caller-supplied copy never reaches
/// the wire next to the computed one.
pub fn is_sigv4_computed_header(name: &str) -> bool {
    SIGV4_COMPUTED_HEADER_NAMES.contains(&name.to_ascii_lowercase().as_str())
}

#[derive(Clone, Copy, Debug)]
pub struct SigningScope<'a> {
    pub region: &'a str,
    pub service: &'a str,
    pub credentials: &'a Credentials,
    pub time: SystemTime,
}

pub fn sign_headers(
    method: &str,
    url: &str,
    body: &[u8],
    headers: &HeaderMap,
    scope: SigningScope<'_>,
) -> Result<HeaderMap, Error> {
    let header_refs = headers
        .iter()
        .map(|(name, value)| {
            value
                .to_str()
                .map(|value| (name.as_str(), value))
                .map_err(|_| Error::NonAsciiHeader {
                    name: name.to_string(),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let instructions = sign_request(
        method,
        url,
        header_refs.into_iter(),
        body,
        SigningSettings::default(),
        scope,
    )?;
    instructions
        .headers()
        .map(|(name, value)| {
            let name = HeaderName::from_bytes(name.as_bytes())
                .map_err(Error::signing(SigningStage::Output))?;
            let value =
                HeaderValue::from_str(value).map_err(Error::signing(SigningStage::Output))?;
            Ok((name, value))
        })
        .collect()
}

pub fn presign_url(
    url: &str,
    expires_in: Duration,
    scope: SigningScope<'_>,
) -> Result<String, Error> {
    let mut signed = url::Url::parse(url).map_err(Error::signing(SigningStage::Request))?;
    let mut settings = SigningSettings::default();
    settings.signature_location = SignatureLocation::QueryParams;
    settings.expires_in = Some(expires_in);
    let instructions = sign_request(
        "GET",
        signed.as_str(),
        std::iter::empty(),
        &[],
        settings,
        scope,
    )?;
    let (signature, auth): (Vec<_>, Vec<_>) = instructions
        .params()
        .iter()
        .map(|(name, value)| (*name, value.as_ref()))
        .partition(|(name, _)| *name == "X-Amz-Signature");
    signed
        .query_pairs_mut()
        .extend_pairs(auth.into_iter().chain(signature));
    Ok(signed.into())
}

fn sign_request<'a>(
    method: &'a str,
    url: &'a str,
    headers: impl Iterator<Item = (&'a str, &'a str)>,
    body: &'a [u8],
    settings: SigningSettings,
    scope: SigningScope<'_>,
) -> Result<SigningInstructions, Error> {
    let identity: Identity = scope.credentials.clone().into();
    let params = v4::SigningParams::builder()
        .identity(&identity)
        .region(scope.region)
        .name(scope.service)
        .time(scope.time)
        .settings(settings)
        .build()
        .map(SigningParams::from)
        .map_err(Error::signing(SigningStage::Parameters))?;
    let request = SignableRequest::new(method, url, headers, SignableBody::Bytes(body))
        .map_err(Error::signing(SigningStage::Request))?;
    let (instructions, _) = sign(request, &params)
        .map_err(Error::signing(SigningStage::Signature))?
        .into_parts();
    Ok(instructions)
}

/// Model-id and region parsing shared by every Bedrock route.
pub fn bedrock_model_id_and_region(model: &str) -> (String, Option<String>) {
    let mut stripped = model;
    for prefix in ["bedrock/converse/", "bedrock/", "converse/"] {
        if let Some(value) = stripped.strip_prefix(prefix) {
            stripped = value;
            break;
        }
    }
    let mut region = None;
    if let Some((candidate, remainder)) = stripped.split_once('/')
        && is_bedrock_region(candidate)
    {
        region = Some(candidate.to_string());
        stripped = remainder;
    }
    for prefix in ["nova-2/", "nova/"] {
        if let Some(value) = stripped.strip_prefix(prefix) {
            stripped = value;
            break;
        }
    }
    if region.is_none() {
        // Python splits the whole ARN and takes field 3, the region. Stripping
        // `arn:` first shifts every field down one, so the region is field 2
        // here; field 3 is the account id.
        region = stripped
            .strip_prefix("arn:")
            .and_then(|value| value.split(':').nth(2))
            .filter(|value| !value.is_empty())
            .map(str::to_string);
    }
    (stripped.to_string(), region)
}

fn is_bedrock_region(value: &str) -> bool {
    value.len() > 3
        && value.contains('-')
        && value
            .chars()
            .all(|char| char.is_ascii_alphanumeric() || char == '-')
}

/// The region a caller configured: `aws_region_name`, then the model's own
/// region, then the environment. Each service decides what a missing one means.
pub fn resolve_aws_region(
    model_region: Option<&str>,
    params: &AwsParams,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    params
        .resolve(&AwsParams::REGION, &|_| None)
        .or_else(|| model_region.map(str::to_string))
        .or_else(|| AwsParams::REGION.resolve(&|_| None, env_lookup))
}

pub fn resolve_bedrock_region(
    model_region: Option<&str>,
    params: &AwsParams,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    bedrock_region(model_region, params, env_lookup, &|name| {
        std::env::var(name).ok()
    })
}

fn bedrock_region(
    model_region: Option<&str>,
    params: &AwsParams,
    env_lookup: &dyn Fn(&str) -> Option<String>,
    process_env: &dyn Fn(&str) -> Option<String>,
) -> String {
    resolve_aws_region(model_region, params, env_lookup)
        .or_else(|| session_region(env_lookup, process_env))
        .unwrap_or_else(|| DEFAULT_BEDROCK_REGION.to_string())
}

fn configured(
    params: &AwsParams,
    spec: &ParamSpec,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    spec.wire
        .iter()
        .find_map(|name| params.get(name))
        .or_else(|| spec.env.iter().find_map(|name| env_lookup(name)))
}

pub fn aws_auth_config(
    params: &AwsParams,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> AwsAuthConfig {
    let secret = |spec: &ParamSpec| configured(params, spec, env_lookup).map(SecretValue::new);
    let setting = |spec: &ParamSpec| configured(params, spec, env_lookup);
    AwsAuthConfig {
        access_key_id: secret(&AwsParams::ACCESS_KEY_ID),
        secret_access_key: secret(&AwsParams::SECRET_ACCESS_KEY),
        session_token: secret(&AwsParams::SESSION_TOKEN),
        region_name: setting(&AwsParams::REGION),
        session_name: setting(&AwsParams::SESSION_NAME),
        profile_name: setting(&AwsParams::PROFILE_NAME),
        role_name: setting(&AwsParams::ROLE_NAME),
        web_identity_token: secret(&AwsParams::WEB_IDENTITY_TOKEN),
        sts_endpoint: setting(&AwsParams::STS_ENDPOINT),
        external_id: secret(&AwsParams::EXTERNAL_ID),
        session_tags: params.aws_session_tags.clone().map(|mut tags| {
            tags.sort_by(|left, right| left.key.cmp(&right.key));
            tags
        }),
        session_policy: None,
    }
}

pub fn web_identity_session_policy(service: &str) -> Option<&'static str> {
    matches!(
        service,
        "bedrock" | "bedrock-agentcore" | "bedrock-mantle" | "aws-external-anthropic"
    )
    .then_some(BEDROCK_SESSION_POLICY)
}

/// Where the credentials that sign a request come from, decided when the request is
/// prepared and resolved when it is sent.
#[derive(Clone, Debug, PartialEq)]
pub enum AwsCredentialSource {
    HostSupplied(Credentials),
    Chain(Box<AwsAuthConfig>),
}

impl AwsCredentialSource {
    pub fn from_params(params: &AwsParams, env_lookup: &dyn Fn(&str) -> Option<String>) -> Self {
        match host_supplied_credentials(params) {
            Some(credentials) if names_only_static_keys(params) => Self::HostSupplied(credentials),
            _ => Self::Chain(Box::new(aws_auth_config(params, env_lookup))),
        }
    }

    pub async fn resolve(
        self,
        auth: &AwsAuthService,
        region: &str,
        service: &str,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Credentials, Error> {
        let config = match self {
            Self::HostSupplied(credentials) => AwsAuthConfig {
                access_key_id: Some(SecretValue::new(credentials.access_key_id())),
                secret_access_key: Some(SecretValue::new(credentials.secret_access_key())),
                session_token: credentials.session_token().map(SecretValue::new),
                ..AwsAuthConfig::default()
            },
            Self::Chain(config) => *config,
        };
        auth.resolve_credentials(
            AwsAuthConfig {
                region_name: Some(region.to_owned()),
                session_policy: config
                    .session_policy
                    .or_else(|| web_identity_session_policy(service)),
                ..config
            },
            env_lookup,
        )
        .await
    }
}

fn names_only_static_keys(params: &AwsParams) -> bool {
    let AwsParams {
        aws_session_token,
        aws_session_name,
        aws_profile_name,
        aws_role_name,
        aws_web_identity_token,
        aws_sts_endpoint,
        aws_external_id,
        aws_session_tags,
        ..
    } = params;
    aws_session_token
        .as_deref()
        .is_none_or(|token| !token.trim().is_empty())
        && [
            aws_session_name,
            aws_profile_name,
            aws_role_name,
            aws_web_identity_token,
            aws_sts_endpoint,
            aws_external_id,
        ]
        .iter()
        .all(|value| value.is_none())
        && aws_session_tags.is_none()
}

pub fn host_supplied_credentials(params: &AwsParams) -> Option<Credentials> {
    let value = |value: &Option<String>| {
        value
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    };
    let access_key_id = value(&params.aws_access_key_id)?;
    let secret_access_key = value(&params.aws_secret_access_key)?;
    Some(Credentials::new(
        access_key_id,
        secret_access_key,
        value(&params.aws_session_token),
        None,
        "litellm-host-supplied",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{AWS_REGION, BEDROCK_SERVICE};
    use reqwest::header::CONTENT_TYPE;

    #[cfg(unix)]
    #[rstest::fixture]
    fn process_profile() -> aws_runtime::env_config::file::EnvConfigFiles {
        let credentials = serde_json::json!({
            "Version": 1,
            "AccessKeyId": "process-access",
            "SecretAccessKey": "process-secret",
            "SessionToken": "process-session",
        });
        aws_runtime::env_config::file::EnvConfigFiles::builder()
            .include_default_config_file(false)
            .include_default_credentials_file(false)
            .with_contents(
                aws_runtime::env_config::file::EnvConfigFileKind::Config,
                format!(
                    "[profile smoke]\ncredential_process = /usr/bin/printf '%s' '{credentials}'\n"
                ),
            )
            .build()
    }

    #[cfg(unix)]
    #[rstest::rstest]
    #[tokio::test]
    async fn profile_process_credentials_can_be_loaded(
        process_profile: aws_runtime::env_config::file::EnvConfigFiles,
    ) {
        let provider = profile_provider(
            "smoke",
            Some("us-west-2".into()),
            aws_config::provider_config::ProviderConfig::without_region(),
            aws_config::profile::ProfileFileCredentialsProvider::builder()
                .profile_files(process_profile),
        )
        .await
        .unwrap();
        let credentials = provider.provide_credentials().await.unwrap();

        assert_eq!(credentials.access_key_id(), "process-access");
        assert_eq!(credentials.secret_access_key(), "process-secret");
        assert_eq!(credentials.session_token(), Some("process-session"));
    }

    #[derive(Debug)]
    struct StsConnector;

    impl aws_smithy_runtime_api::client::http::HttpConnector for StsConnector {
        fn call(
            &self,
            request: aws_smithy_runtime_api::client::orchestrator::HttpRequest,
        ) -> aws_smithy_runtime_api::client::http::HttpConnectorFuture {
            assert!(request.uri().contains("sts.us-west-2."));
            aws_smithy_runtime_api::client::http::HttpConnectorFuture::new(async {
                Ok(
                    aws_smithy_runtime_api::client::orchestrator::HttpResponse::new(
                        200_u16.try_into().unwrap(),
                        aws_smithy_types::body::SdkBody::from(
                            r#"<AssumeRoleResponse xmlns="https://sts.amazonaws.com/doc/2011-06-15/">
                        <AssumeRoleResult><Credentials>
                        <AccessKeyId>role-access</AccessKeyId><SecretAccessKey>role-secret</SecretAccessKey>
                        <SessionToken>role-session</SessionToken><Expiration>2100-01-01T00:00:00Z</Expiration>
                        </Credentials><AssumedRoleUser><Arn>arn:aws:sts::123456789012:assumed-role/smoke/session</Arn>
                        <AssumedRoleId>smoke:session</AssumedRoleId></AssumedRoleUser></AssumeRoleResult>
                        <ResponseMetadata><RequestId>smoke</RequestId></ResponseMetadata></AssumeRoleResponse>"#,
                        ),
                    ),
                )
            })
        }
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn role_profile_uses_region_when_acquiring_sts_credentials() {
        use aws_smithy_runtime_api::client::http::{SharedHttpConnector, http_client_fn};
        let files = aws_runtime::env_config::file::EnvConfigFiles::builder()
            .include_default_config_file(false)
            .include_default_credentials_file(false)
            .with_contents(
                aws_runtime::env_config::file::EnvConfigFileKind::Config,
                "[profile smoke]\nrole_arn = arn:aws:iam::123456789012:role/smoke\nsource_profile = source\n\n[profile source]\naws_access_key_id = source-access\naws_secret_access_key = source-secret\n",
            ).build();
        let config = aws_config::provider_config::ProviderConfig::without_region()
            .with_http_client(http_client_fn(|_, _| {
                SharedHttpConnector::new(StsConnector)
            }));
        let credentials = profile_provider(
            "smoke",
            Some("us-west-2".into()),
            config,
            aws_config::profile::ProfileFileCredentialsProvider::builder().profile_files(files),
        )
        .await
        .unwrap();
        let credentials = credentials.provide_credentials().await.unwrap();
        assert_eq!(credentials.access_key_id(), "role-access");
        assert_eq!(credentials.secret_access_key(), "role-secret");
        assert_eq!(credentials.session_token(), Some("role-session"));
        assert!(credentials.expiry().is_some());
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[rstest::rstest]
    #[case::access_key(Some("source-access"), None)]
    #[case::secret_key(None, Some("source-secret"))]
    #[case::source_pair(Some("source-access"), Some("source-secret"))]
    #[tokio::test]
    async fn explicit_source_keys_cannot_reuse_an_ambient_role(
        #[case] access_key: Option<&str>,
        #[case] secret_key: Option<&str>,
    ) {
        let role = "arn:aws:iam::123456789012:role/target";
        let running_as = RunningAs {
            role: role.to_string(),
            config: AwsAuthConfig {
                access_key_id: access_key.map(SecretValue::new),
                secret_access_key: secret_key.map(SecretValue::new),
                ..Default::default()
            },
            sts_region: None,
            loader: NativeProviderLoader::default(),
        };
        assert!(!running_as.is_target(Some(role)).await);
    }

    #[test]
    fn secret_names_cover_environment_reads() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(
            std::collections::BTreeSet::<String>::new(),
        ));
        let recorded = seen.clone();
        let env = |name: &str| {
            recorded.lock().unwrap().insert(name.to_string());
            None
        };
        bedrock_region(None, &AwsParams::default(), &env, &|_| None);
        aws_auth_config(&AwsParams::default(), &env);
        assert!(
            seen.lock()
                .unwrap()
                .iter()
                .all(|name| AwsParams::secret_names().contains(&name.as_str()))
        );
    }

    #[rstest::rstest]
    #[case::param_wins(Some("from-params"), &[(AWS_SECRET_ACCESS_KEY, "from-env")], Some("from-params"))]
    #[case::blank_param_counts_as_set(Some("  "), &[(AWS_SECRET_ACCESS_KEY, "from-env")], Some("  "))]
    #[case::blank_environment_counts_as_set(None, &[(AWS_SECRET_ACCESS_KEY, "")], Some(""))]
    #[case::absent_param_falls_to_environment(None, &[(AWS_SECRET_ACCESS_KEY, "from-env")], Some("from-env"))]
    #[case::nothing(None, &[], None)]
    fn auth_config_reads_each_spec_from_params_then_the_environment(
        #[case] aws_secret_access_key: Option<&str>,
        #[case] environment: &[(&str, &str)],
        #[case] expected: Option<&str>,
    ) {
        let params = AwsParams {
            aws_secret_access_key: aws_secret_access_key.map(str::to_string),
            ..AwsParams::default()
        };
        let env = |key: &str| {
            environment
                .iter()
                .find(|(name, _)| *name == key)
                .map(|(_, value)| value.to_string())
        };

        assert_eq!(
            aws_auth_config(&params, &env)
                .secret_access_key
                .as_ref()
                .map(SecretValue::expose),
            expected
        );
        assert_eq!(
            aws_auth_config(&AwsParams::default(), &|key: &str| (key == AWS_REGION)
                .then(|| "sa-east-1".to_string()))
            .region_name
            .as_deref(),
            Some("sa-east-1")
        );
    }

    #[rstest::rstest]
    #[case::call_params(Some("us-east-2"), Some("eu-west-1"), &[(AWS_REGION_NAME, "ap-south-1")], Some("eu-west-1"))]
    #[case::model_region(Some("us-east-2"), None, &[(AWS_REGION_NAME, "ap-south-1")], Some("us-east-2"))]
    #[case::aws_region_name(None, None, &[(AWS_REGION_NAME, "ap-south-1")], Some("ap-south-1"))]
    #[case::aws_region(None, None, &[(AWS_REGION, "sa-east-1"), (AWS_DEFAULT_REGION, "eu-north-1")], Some("sa-east-1"))]
    #[case::nothing(None, None, &[], None)]
    fn a_region_comes_from_the_call_then_the_model_then_the_environment(
        #[case] model_region: Option<&str>,
        #[case] aws_region_name: Option<&str>,
        #[case] environment: &[(&str, &str)],
        #[case] expected: Option<&str>,
    ) {
        let params = AwsParams {
            aws_region_name: aws_region_name.map(str::to_string),
            ..AwsParams::default()
        };
        let env = |key: &str| {
            environment
                .iter()
                .find(|(name, _)| *name == key)
                .map(|(_, value)| value.to_string())
        };

        assert_eq!(
            resolve_aws_region(model_region, &params, &env).as_deref(),
            expected
        );
        assert_eq!(
            bedrock_region(model_region, &params, &env, &|_| None),
            expected.unwrap_or(DEFAULT_BEDROCK_REGION)
        );
    }

    #[rstest::rstest]
    #[case::default_region(&[(AWS_DEFAULT_REGION, "eu-north-1")], "eu-north-1")]
    #[case::blank_default_region(&[(AWS_DEFAULT_REGION, "")], DEFAULT_BEDROCK_REGION)]
    #[case::aws_region_outranks_it(&[(AWS_DEFAULT_REGION, "eu-north-1"), (AWS_REGION, "sa-east-1")], "sa-east-1")]
    fn bedrock_falls_back_to_the_boto3_session_region(
        #[case] environment: &[(&str, &str)],
        #[case] expected: &str,
    ) {
        let env = |key: &str| {
            environment
                .iter()
                .find(|(name, _)| *name == key)
                .map(|(_, value)| value.to_string())
        };

        assert_eq!(
            bedrock_region(None, &AwsParams::default(), &env, &|_| None),
            expected
        );
    }

    fn parity_inputs() -> (String, Vec<u8>, HeaderMap) {
        (
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.titan-text-express-v1/invoke"
                .to_string(),
            br#"{"input":"hello"}"#.to_vec(),
            HeaderMap::from_iter([(CONTENT_TYPE, HeaderValue::from_static("application/json"))]),
        )
    }

    #[test]
    fn reads_the_region_field_of_a_model_arn_not_the_account_id() {
        // Python's `_get_aws_region_from_model_arn` splits the whole ARN and
        // takes field 3. Stripping `arn:` first shifts every field down one, so
        // the region is field 2 here. Taking field 3 after the strip returns
        // the account id, which is not a region at all.
        let (_, region) = bedrock_model_id_and_region(
            "bedrock/arn:aws:bedrock:us-west-2:123456789012:foundation-model/anthropic.claude-v2",
        );
        assert_eq!(region.as_deref(), Some("us-west-2"));
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

    #[rstest::rstest]
    #[case::running_with_web_identity("/var/run/token", Some("arn:aws:iam::1:role/current"))]
    #[case::empty_token_file("", None)]
    #[tokio::test]
    async fn assume_role_reads_the_ambient_role_from_the_secret_snapshot(
        #[case] token_file: &str,
        #[case] expected: Option<&str>,
    ) {
        use litellm_secrets::source::SecretSource;

        struct WebIdentitySecrets<'a>(&'a str);

        impl SecretSource for WebIdentitySecrets<'_> {
            fn get_secret_str<'a>(
                &'a self,
                name: &'a str,
            ) -> futures_util::future::BoxFuture<
                'a,
                Result<Option<SecretValue>, litellm_secrets::Error>,
            > {
                Box::pin(async move {
                    Ok(match name {
                        AWS_WEB_IDENTITY_TOKEN_FILE => Some(SecretValue::new(self.0)),
                        AWS_ROLE_ARN => Some(SecretValue::new("arn:aws:iam::1:role/current")),
                        _ => None,
                    })
                })
            }
        }

        let snapshot = WebIdentitySecrets(token_file)
            .resolve(AwsParams::secret_names())
            .await
            .unwrap();
        let config = AwsAuthConfig {
            role_name: Some("arn:aws:iam::1:role/target".into()),
            ..Default::default()
        };
        let AwsAuthFlow::AssumeRole {
            web_identity_role, ..
        } = classify_auth(config, &|name| snapshot.get(name))
        else {
            panic!("a role name selects AssumeRole");
        };
        assert_eq!(web_identity_role.as_deref(), expected);
    }

    #[tokio::test]
    async fn static_credentials_do_not_use_network() {
        let credentials = AwsAuthService::default()
            .resolve_credentials(
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

    struct CountingLoader {
        loads: Arc<std::sync::atomic::AtomicUsize>,
        calls: Arc<std::sync::atomic::AtomicUsize>,
        short_lifetime: Duration,
        short_calls: usize,
    }

    impl ProviderLoader for CountingLoader {
        fn load(&self, config: AwsAuthConfig, _: AwsAuthFlow) -> ProviderFuture<'_> {
            let loads = self.loads.clone();
            let provider = CountingProvider {
                calls: self.calls.clone(),
                short_lifetime: self.short_lifetime,
                short_calls: self.short_calls,
                key: config.access_key_id.unwrap(),
            };
            Box::pin(async move {
                loads.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(SharedCredentialsProvider::new(provider))
            })
        }
    }

    #[derive(Debug)]
    struct CountingProvider {
        calls: Arc<std::sync::atomic::AtomicUsize>,
        short_lifetime: Duration,
        short_calls: usize,
        key: SecretValue,
    }

    impl ProvideCredentials for CountingProvider {
        fn provide_credentials<'a>(
            &'a self,
        ) -> aws_credential_types::provider::future::ProvideCredentials<'a>
        where
            Self: 'a,
        {
            aws_credential_types::provider::future::ProvideCredentials::new(async move {
                tokio::task::yield_now().await;
                let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let lifetime = if call < self.short_calls {
                    self.short_lifetime
                } else {
                    Duration::from_secs(3600)
                };
                Ok(Credentials::new(
                    self.key.expose(),
                    "credential-value",
                    Some(format!("session-{call}")),
                    Some(SystemTime::now() + lifetime),
                    "counting",
                ))
            })
        }
    }

    #[derive(Default)]
    struct RecordingLoader {
        loads: std::sync::Mutex<Vec<(AwsAuthConfig, AwsAuthFlow)>>,
    }

    impl ProviderLoader for RecordingLoader {
        fn load(&self, config: AwsAuthConfig, flow: AwsAuthFlow) -> ProviderFuture<'_> {
            self.loads.lock().unwrap().push((config, flow));
            Box::pin(async {
                Ok(SharedCredentialsProvider::new(Credentials::new(
                    "ak",
                    "sk",
                    None,
                    None,
                    "recording",
                )))
            })
        }
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn default_chain_credentials_do_not_share_a_configured_role_in_the_same_region() {
        let loader = Arc::new(RecordingLoader::default());
        let auth = AwsAuthService::new(loader.clone());
        let bedrock_env = |name: &str| match name {
            AWS_ROLE_NAME => Some("arn:aws:iam::123456789012:role/bedrock".to_string()),
            AWS_REGION_NAME => Some("us-east-1".to_string()),
            _ => None,
        };
        auth.resolve_credentials(AwsAuthConfig::default(), &bedrock_env)
            .await
            .unwrap();

        auth.default_chain_credentials("us-east-1").await.unwrap();
        auth.default_chain_credentials("us-east-1").await.unwrap();

        let loads = loader.loads.lock().unwrap();
        assert_eq!(loads.len(), 2);
        assert_eq!(
            loads[1],
            (
                AwsAuthConfig {
                    region_name: Some("us-east-1".to_string()),
                    ..Default::default()
                },
                AwsAuthFlow::DefaultChain
            )
        );
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn near_expiry_credentials_are_refreshed_without_rebuilding_the_provider() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let loads = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(AtomicUsize::new(0));
        let auth = AwsAuthService::new(Arc::new(CountingLoader {
            loads: loads.clone(),
            calls: calls.clone(),
            short_lifetime: Duration::from_secs(30),
            short_calls: 1,
        }));
        let config = AwsAuthConfig {
            access_key_id: Some("identity".into()),
            ..Default::default()
        };
        let first = auth
            .resolve_credentials(config.clone(), &no_env)
            .await
            .unwrap();
        let second = auth
            .clone()
            .resolve_credentials(config.clone(), &no_env)
            .await
            .unwrap();
        assert_eq!(first.session_token(), Some("session-1"));
        assert_eq!(first.session_token(), second.session_token());
        assert_eq!(loads.load(Ordering::SeqCst), 1);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[rstest::rstest]
    #[case::recovered_by_one_refetch(1, Some("session-1"))]
    #[case::still_expired_after_refetch(2, None)]
    #[tokio::test]
    async fn expired_credentials_are_refetched_once(
        #[case] short_calls: usize,
        #[case] session: Option<&str>,
    ) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = Arc::new(AtomicUsize::new(0));
        let auth = AwsAuthService::new(Arc::new(CountingLoader {
            loads: Arc::new(AtomicUsize::new(0)),
            calls: calls.clone(),
            short_lifetime: Duration::ZERO,
            short_calls,
        }));
        let config = AwsAuthConfig {
            access_key_id: Some("identity".into()),
            ..Default::default()
        };
        let result = auth.resolve_credentials(config, &no_env).await;
        assert_eq!(
            result.as_ref().ok().and_then(Credentials::session_token),
            session
        );
        assert!(session.is_some() || matches!(result, Err(Error::ExpiredCredentials)));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[rstest::rstest]
    #[tokio::test]
    async fn concurrent_calls_share_acquisition_and_distinct_identities_stay_isolated() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let loads = Arc::new(AtomicUsize::new(0));
        let calls = Arc::new(AtomicUsize::new(0));
        let auth = AwsAuthService::new(Arc::new(CountingLoader {
            loads: loads.clone(),
            calls: calls.clone(),
            short_lifetime: Duration::from_secs(3600),
            short_calls: 0,
        }));
        let first = AwsAuthConfig {
            access_key_id: Some("first".into()),
            ..Default::default()
        };
        let second = AwsAuthConfig {
            access_key_id: Some("second".into()),
            ..Default::default()
        };
        let (a, b, c) = tokio::join!(
            auth.resolve_credentials(first.clone(), &no_env),
            auth.resolve_credentials(first, &no_env),
            auth.resolve_credentials(second, &no_env)
        );
        let a = a.unwrap();
        let b = b.unwrap();
        let c = c.unwrap();
        assert_eq!(a.session_token(), b.session_token());
        assert_ne!(a.access_key_id(), c.access_key_id());
        assert_eq!(loads.load(Ordering::SeqCst), 2);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[rstest::rstest]
    fn auth_config_debug_never_discloses_credentials() {
        let config = AwsAuthConfig {
            access_key_id: Some("credential-value".into()),
            secret_access_key: Some("credential-value".into()),
            session_token: Some("credential-value".into()),
            web_identity_token: Some("credential-value".into()),
            external_id: Some("credential-value".into()),
            ..Default::default()
        };
        let output = format!("{config:?}");
        assert!(!output.contains("credential-value"));
        assert!(output.contains("[REDACTED]"));
    }

    #[rstest::rstest]
    #[case::session(AwsAuthFlow::SessionToken { access_key_id: "credential-value".into(), secret_access_key: "credential-value".into(), session_token: "credential-value".into() })]
    #[case::static_keys(AwsAuthFlow::StaticKeys { access_key_id: "credential-value".into(), secret_access_key: "credential-value".into(), region_name: "region".into() })]
    #[case::web_identity(AwsAuthFlow::WebIdentity { token: "credential-value".into(), role: "role".into(), session_name: "session".into(), sts_region: None })]
    fn auth_flow_debug_never_discloses_credentials(#[case] flow: AwsAuthFlow) {
        let output = format!("{flow:?}");
        assert!(!output.contains("credential-value"));
        assert!(output.contains("[REDACTED]"));
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
    fn a_forwarded_client_header_is_not_folded_into_the_signature() {
        // Python signs only the AWS header set, so a header a caller forwarded
        // cannot change the canonical request. Signing it instead makes the
        // request 403 the moment anything on the wire rewrites or drops it.
        let (url, body, mut headers) = parity_inputs();
        headers.insert("x-request-id", HeaderValue::from_static("abc-123"));
        headers.insert("Accept-Encoding", HeaderValue::from_static("gzip"));
        headers.insert("x-amzn-trace-id", HeaderValue::from_static("Root=1-abc"));
        let signable = aws_signature_headers(&headers);

        assert!(!signable.contains_key("x-request-id"));
        assert!(!signable.contains_key("Accept-Encoding"));
        // The AWS-prefixed one is genuinely part of the signature.
        assert!(signable.contains_key("x-amzn-trace-id"));
        assert!(signable.contains_key("Content-Type"));

        let credentials = Credentials::new(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            None,
            None,
            "test",
        );
        let signed = sign_headers(
            "POST",
            &url,
            &body,
            &signable,
            SigningScope {
                region: "us-east-1",
                service: BEDROCK_SERVICE,
                credentials: &credentials,
                time: SystemTime::UNIX_EPOCH,
            },
        )
        .expect("signs");
        let authorization = signed["Authorization"]
            .to_str()
            .expect("carries an authorization header");
        assert!(
            !authorization.contains("x-request-id"),
            "forwarded header reached SignedHeaders: {authorization}"
        );
        assert!(
            !authorization.contains("accept-encoding"),
            "forwarded header reached SignedHeaders: {authorization}"
        );
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
        let signed = sign_headers(
            "POST",
            &url,
            &body,
            &headers,
            SigningScope {
                region: "us-east-1",
                service: BEDROCK_SERVICE,
                credentials: &credentials,
                time: UNIX_EPOCH + std::time::Duration::from_secs(1_704_164_645),
            },
        )
        .expect("golden signature");
        assert_eq!(signed["X-Amz-Date"], "20240102T030405Z");
        assert_eq!(signed["X-Amz-Security-Token"], "session-token");
        assert_eq!(
            signed["Authorization"],
            "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20240102/us-east-1/bedrock/aws4_request, SignedHeaders=content-type;host;x-amz-date;x-amz-security-token, Signature=55c027ef47527d3ad63f1735f9d099efdbc99f296ff914bd94e727e24ec0e464"
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
        let signed = sign_headers(
            "POST",
            &url,
            &body,
            &headers,
            SigningScope {
                region: "us-east-1",
                service: BEDROCK_SERVICE,
                credentials: &credentials,
                time: UNIX_EPOCH + std::time::Duration::from_secs(1_704_164_645),
            },
        )
        .expect("signature");
        assert!(!signed.contains_key("X-Amz-Security-Token"));
    }

    #[test]
    fn a_header_value_outside_visible_ascii_is_named_instead_of_signed() {
        let (url, body, mut headers) = parity_inputs();
        headers.insert("x-amz-meta-note", HeaderValue::from_str("café").unwrap());
        let credentials = Credentials::new("AKIDEXAMPLE", "secret", None, None, "test");

        let error = sign_headers(
            "POST",
            &url,
            &body,
            &headers,
            SigningScope {
                region: "us-east-1",
                service: BEDROCK_SERVICE,
                credentials: &credentials,
                time: UNIX_EPOCH,
            },
        )
        .unwrap_err();

        assert_eq!(
            error,
            Error::NonAsciiHeader {
                name: "x-amz-meta-note".into()
            }
        );
    }

    #[test]
    fn the_shared_auth_error_nests_the_aws_failure_and_its_cause() {
        let cause = std::io::Error::new(std::io::ErrorKind::NotFound, "profile not found");
        let error =
            litellm_auth_types::Error::from(Error::credentials(CredentialFlow::Profile)(cause));

        assert!(matches!(
            litellm_error::find::<Error>(&error),
            Some(Error::Credentials {
                flow: CredentialFlow::Profile,
                ..
            })
        ));
        let report = litellm_error::Report::new(&error).to_string();
        assert!(
            report.contains("could not load AWS profile credentials"),
            "{report}"
        );
        assert_eq!(report.matches("profile not found").count(), 1, "{report}");
    }

    #[derive(Clone, Debug, Default)]
    struct Sts {
        requests: Arc<std::sync::Mutex<Vec<StsRequest>>>,
        deny_assume_role: bool,
        caller_arns: Arc<std::sync::Mutex<Vec<Option<&'static str>>>>,
    }

    #[derive(Clone, Debug)]
    struct StsRequest {
        host: String,
        authorization: Option<String>,
        form: std::collections::BTreeMap<String, String>,
    }

    impl StsRequest {
        fn action(&self) -> &str {
            self.form["Action"].as_str()
        }
    }

    const STS_XMLNS: &str = "https://sts.amazonaws.com/doc/2011-06-15/";

    impl Sts {
        fn response(&self, request: &StsRequest) -> (u16, String) {
            let credentials = |prefix: &str| {
                format!(
                    "<Credentials><AccessKeyId>{prefix}-access</AccessKeyId><SecretAccessKey>{prefix}-secret</SecretAccessKey><SessionToken>{prefix}-session</SessionToken><Expiration>2100-01-01T00:00:00Z</Expiration></Credentials><AssumedRoleUser><Arn>arn:aws:sts::123456789012:assumed-role/{prefix}/s</Arn><AssumedRoleId>id:s</AssumedRoleId></AssumedRoleUser>"
                )
            };
            let error = |code: &str| {
                format!(
                    "<ErrorResponse xmlns=\"{STS_XMLNS}\"><Error><Type>Sender</Type><Code>{code}</Code><Message>denied</Message></Error><RequestId>r</RequestId></ErrorResponse>"
                )
            };
            match request.action() {
                "AssumeRole" if self.deny_assume_role => (403, error("AccessDenied")),
                "AssumeRole" => (
                    200,
                    format!(
                        "<AssumeRoleResponse xmlns=\"{STS_XMLNS}\"><AssumeRoleResult>{}</AssumeRoleResult></AssumeRoleResponse>",
                        credentials("role")
                    ),
                ),
                "AssumeRoleWithWebIdentity" => (
                    200,
                    format!(
                        "<AssumeRoleWithWebIdentityResponse xmlns=\"{STS_XMLNS}\"><AssumeRoleWithWebIdentityResult>{}</AssumeRoleWithWebIdentityResult></AssumeRoleWithWebIdentityResponse>",
                        credentials("web")
                    ),
                ),
                "GetCallerIdentity" => match self.caller_arns.lock().unwrap().remove(0) {
                    Some(arn) => (
                        200,
                        format!(
                            "<GetCallerIdentityResponse xmlns=\"{STS_XMLNS}\"><GetCallerIdentityResult><Arn>{arn}</Arn><UserId>u</UserId><Account>123456789012</Account></GetCallerIdentityResult></GetCallerIdentityResponse>"
                        ),
                    ),
                    None => (400, error("InvalidClientTokenId")),
                },
                action => panic!("unexpected STS action {action}"),
            }
        }

        fn sent(&self) -> Vec<StsRequest> {
            self.requests.lock().unwrap().clone()
        }
    }

    impl aws_smithy_runtime_api::client::http::HttpConnector for Sts {
        fn call(
            &self,
            request: aws_smithy_runtime_api::client::orchestrator::HttpRequest,
        ) -> aws_smithy_runtime_api::client::http::HttpConnectorFuture {
            let recorded = StsRequest {
                host: url::Url::parse(request.uri())
                    .unwrap()
                    .host_str()
                    .unwrap()
                    .to_string(),
                authorization: request.headers().get("authorization").map(str::to_string),
                form: url::form_urlencoded::parse(request.body().bytes().unwrap_or_default())
                    .into_owned()
                    .collect(),
            };
            let (status, body) = self.response(&recorded);
            self.requests.lock().unwrap().push(recorded);
            aws_smithy_runtime_api::client::http::HttpConnectorFuture::new(async move {
                Ok(
                    aws_smithy_runtime_api::client::orchestrator::HttpResponse::new(
                        status.try_into().unwrap(),
                        aws_smithy_types::body::SdkBody::from(body),
                    ),
                )
            })
        }
    }

    #[derive(Debug)]
    struct Tokens(Option<&'static str>);

    impl WebIdentityTokenSource for Tokens {
        fn web_identity_token<'a>(&'a self, reference: &'a str) -> WebIdentityTokenFuture<'a> {
            Box::pin(async move {
                assert_eq!(reference, "oidc/file/var/run/secrets/token");
                Ok(self.0.map(SecretValue::new))
            })
        }
    }

    fn sts_service(sts: &Sts, tokens: Option<Arc<dyn WebIdentityTokenSource>>) -> AwsAuthService {
        use aws_smithy_runtime_api::client::http::{SharedHttpConnector, http_client_fn};
        let connector = sts.clone();
        AwsAuthService::new(Arc::new(NativeProviderLoader {
            tokens,
            http_client: Some(SharedHttpClient::new(http_client_fn(move |_, _| {
                SharedHttpConnector::new(connector.clone())
            }))),
            ambient: Some(SharedCredentialsProvider::new(Credentials::new(
                "ambient-access",
                "ambient-secret",
                None,
                None,
                "ambient",
            ))),
        }))
    }

    fn lookup(
        environment: &'static [(&'static str, &'static str)],
    ) -> impl Fn(&str) -> Option<String> + Sync {
        |key: &str| {
            environment
                .iter()
                .find(|(name, _)| *name == key)
                .map(|(_, value)| value.to_string())
        }
    }

    const TARGET_ROLE: &str = "arn:aws:iam::123456789012:role/target";

    #[rstest::rstest]
    #[case::regional_endpoint(Some("https://sts.eu-west-1.amazonaws.com"), &[(AWS_REGION, "ap-south-1")], "sts.eu-west-1.amazonaws.com", "eu-west-1")]
    #[case::aws_region(None, &[(AWS_REGION, "ap-south-1"), (AWS_DEFAULT_REGION, "ca-central-1")], "sts.ap-south-1.amazonaws.com", "ap-south-1")]
    #[case::aws_default_region(None, &[(AWS_DEFAULT_REGION, "ca-central-1")], "sts.ca-central-1.amazonaws.com", "ca-central-1")]
    #[case::the_signing_region(None, &[], "sts.us-east-1.amazonaws.com", "us-east-1")]
    #[case::custom_endpoint_signs_for_the_signing_region(Some("https://sts.internal.test"), &[], "sts.internal.test", "us-east-1")]
    #[tokio::test]
    async fn params_keys_assume_the_role_with_tags_in_the_python_sts_region(
        #[case] sts_endpoint: Option<&str>,
        #[case] environment: &'static [(&'static str, &'static str)],
        #[case] host: &str,
        #[case] sts_region: &str,
    ) {
        let sts = Sts::default();
        let params = AwsParams {
            aws_access_key_id: Some("source-access".into()),
            aws_secret_access_key: Some("source-secret".into()),
            aws_role_name: Some(TARGET_ROLE.into()),
            aws_session_name: Some("attributed".into()),
            aws_sts_endpoint: sts_endpoint.map(str::to_string),
            aws_session_tags: Some(vec![
                AwsSessionTag {
                    key: "team".into(),
                    value: "genai".into(),
                },
                AwsSessionTag {
                    key: "cost".into(),
                    value: "42".into(),
                },
            ]),
            ..AwsParams::default()
        };

        let credentials = AwsCredentialSource::from_params(&params, &no_env)
            .resolve(
                &sts_service(&sts, None),
                "us-east-1",
                BEDROCK_SERVICE,
                &lookup(environment),
            )
            .await
            .unwrap();

        assert_eq!(credentials.access_key_id(), "role-access");
        let [request] = sts.sent().try_into().unwrap();
        assert_eq!(request.action(), "AssumeRole");
        assert_eq!(request.host, host);
        assert!(
            request
                .authorization
                .as_deref()
                .unwrap()
                .contains("Credential=source-access/")
        );
        assert!(
            request
                .authorization
                .unwrap()
                .contains(&format!("/{sts_region}/sts/aws4_request"))
        );
        assert_eq!(request.form["RoleArn"], TARGET_ROLE);
        assert_eq!(request.form["RoleSessionName"], "attributed");
        assert_eq!(
            [
                request.form["Tags.member.1.Key"].as_str(),
                request.form["Tags.member.1.Value"].as_str(),
                request.form["Tags.member.2.Key"].as_str(),
                request.form["Tags.member.2.Value"].as_str(),
            ],
            ["cost", "42", "team", "genai"]
        );
    }

    #[rstest::rstest]
    #[case::caller_is_the_target_role(
        vec![None, Some("arn:aws:sts::123456789012:assumed-role/target/i-1")],
        Ok("ambient-access")
    )]
    #[case::caller_is_another_role(
        vec![None, Some("arn:aws:sts::123456789012:assumed-role/other/i-1")],
        Err(())
    )]
    #[tokio::test]
    async fn an_access_denied_assume_role_falls_back_only_when_already_running_as_the_role(
        #[case] caller_arns: Vec<Option<&'static str>>,
        #[case] expected: Result<&str, ()>,
    ) {
        let sts = Sts {
            deny_assume_role: true,
            caller_arns: Arc::new(std::sync::Mutex::new(caller_arns)),
            ..Sts::default()
        };
        let source = AwsCredentialSource::Chain(Box::new(AwsAuthConfig {
            role_name: Some(TARGET_ROLE.into()),
            ..AwsAuthConfig::default()
        }));

        let result = source
            .resolve(
                &sts_service(&sts, None),
                "us-east-1",
                BEDROCK_SERVICE,
                &no_env,
            )
            .await;

        assert_eq!(
            result
                .as_ref()
                .map(Credentials::access_key_id)
                .map_err(|_| ()),
            expected
        );
        assert_eq!(
            sts.sent()
                .iter()
                .map(StsRequest::action)
                .collect::<Vec<_>>(),
            ["GetCallerIdentity", "AssumeRole", "GetCallerIdentity"]
        );
    }

    fn web_identity_source(token: &str) -> AwsCredentialSource {
        AwsCredentialSource::from_params(
            &AwsParams {
                aws_access_key_id: Some("ignored-access".into()),
                aws_secret_access_key: Some("ignored-secret".into()),
                aws_web_identity_token: Some(token.into()),
                aws_role_name: Some(TARGET_ROLE.into()),
                aws_session_name: Some("oidc-session".into()),
                ..AwsParams::default()
            },
            &no_env,
        )
    }

    #[rstest::rstest]
    #[case::bedrock_carries_the_session_policy(BEDROCK_SERVICE, Some(BEDROCK_SESSION_POLICY))]
    #[case::another_service_does_not("textract", None)]
    #[tokio::test]
    async fn a_web_identity_reference_is_resolved_before_sts_sees_it(
        #[case] service: &'static str,
        #[case] policy: Option<&str>,
    ) {
        let sts = Sts::default();

        let credentials = web_identity_source("oidc/file/var/run/secrets/token")
            .resolve(
                &sts_service(&sts, Some(Arc::new(Tokens(Some("resolved.jwt.value"))))),
                "us-east-1",
                service,
                &no_env,
            )
            .await
            .unwrap();

        assert_eq!(credentials.access_key_id(), "web-access");
        let [request] = sts.sent().try_into().unwrap();
        assert_eq!(request.action(), "AssumeRoleWithWebIdentity");
        assert_eq!(request.form["WebIdentityToken"], "resolved.jwt.value");
        assert_eq!(request.form["RoleArn"], TARGET_ROLE);
        assert_eq!(request.form["RoleSessionName"], "oidc-session");
        assert_eq!(request.form["DurationSeconds"], "3600");
        assert_eq!(request.form.get("Policy").map(String::as_str), policy);
    }

    #[rstest::rstest]
    #[case::environment_prefix("os.environ/OIDC_TOKEN")]
    #[case::environment_name("OIDC_TOKEN")]
    #[tokio::test]
    async fn a_web_identity_reference_to_the_environment_is_refused(#[case] token: &str) {
        let sts = Sts::default();

        let error = web_identity_source(token)
            .resolve(
                &sts_service(&sts, Some(Arc::new(Tokens(Some("unused"))))),
                "us-east-1",
                BEDROCK_SERVICE,
                &lookup(&[("OIDC_TOKEN", "raw.jwt.value")]),
            )
            .await
            .unwrap_err();

        assert_eq!(error, Error::WebIdentityTokenReference);
        assert!(sts.sent().is_empty());
    }

    #[tokio::test]
    async fn a_web_identity_reference_that_resolves_to_nothing_never_reaches_sts() {
        let sts = Sts::default();

        let error = web_identity_source("oidc/file/var/run/secrets/token")
            .resolve(
                &sts_service(&sts, Some(Arc::new(Tokens(None)))),
                "us-east-1",
                BEDROCK_SERVICE,
                &no_env,
            )
            .await
            .unwrap_err();

        assert!(matches!(
            litellm_error::find::<Error>(&error),
            Some(Error::Credentials {
                flow: CredentialFlow::WebIdentity,
                ..
            })
        ));
        assert!(
            litellm_error::Report::new(&error)
                .to_string()
                .contains("resolved to no token")
        );
        assert!(sts.sent().is_empty());
    }

    fn recorded_flow(
        source: AwsCredentialSource,
        environment: &'static [(&'static str, &'static str)],
    ) -> (AwsAuthConfig, AwsAuthFlow) {
        let loader = Arc::new(RecordingLoader::default());
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(source.resolve(
                &AwsAuthService::new(loader.clone()),
                "eu-central-1",
                BEDROCK_SERVICE,
                &lookup(environment),
            ))
            .unwrap();
        loader.loads.lock().unwrap().remove(0)
    }

    fn params_keys(extra: AwsParams) -> AwsParams {
        AwsParams {
            aws_access_key_id: Some("params-access".into()),
            aws_secret_access_key: Some("params-secret".into()),
            ..extra
        }
    }

    #[rstest::rstest]
    #[case::keys_alone_sign_as_static_keys_in_the_signing_region(
        params_keys(AwsParams::default()),
        &[],
        AwsAuthFlow::StaticKeys { access_key_id: "params-access".into(), secret_access_key: "params-secret".into(), region_name: "eu-central-1".into() }
    )]
    #[case::keys_pick_up_an_ambient_session_token(
        params_keys(AwsParams::default()),
        &[(AWS_SESSION_TOKEN, "ambient-token")],
        AwsAuthFlow::SessionToken { access_key_id: "params-access".into(), secret_access_key: "params-secret".into(), session_token: "ambient-token".into() }
    )]
    #[case::keys_with_a_role_assume_it(
        params_keys(AwsParams { aws_role_name: Some(TARGET_ROLE.into()), ..AwsParams::default() }),
        &[],
        AwsAuthFlow::AssumeRole { role: TARGET_ROLE.into(), session_name: None, web_identity_role: None, sts_region: Some("eu-central-1".into()) }
    )]
    #[case::keys_under_an_ambient_role_assume_it(
        params_keys(AwsParams::default()),
        &[(AWS_ROLE_NAME, TARGET_ROLE)],
        AwsAuthFlow::AssumeRole { role: TARGET_ROLE.into(), session_name: None, web_identity_role: None, sts_region: Some("eu-central-1".into()) }
    )]
    #[case::keys_with_a_profile_use_the_profile(
        params_keys(AwsParams { aws_profile_name: Some("work".into()), ..AwsParams::default() }),
        &[],
        AwsAuthFlow::Profile { name: "work".into() }
    )]
    #[case::a_blank_role_is_still_a_role(
        params_keys(AwsParams { aws_role_name: Some(String::new()), ..AwsParams::default() }),
        &[],
        AwsAuthFlow::AssumeRole { role: String::new(), session_name: None, web_identity_role: None, sts_region: Some("eu-central-1".into()) }
    )]
    fn params_keys_follow_python_credential_precedence(
        #[case] params: AwsParams,
        #[case] environment: &'static [(&'static str, &'static str)],
        #[case] expected: AwsAuthFlow,
    ) {
        let (config, flow) = recorded_flow(
            AwsCredentialSource::from_params(&params, &no_env),
            environment,
        );

        assert_eq!(flow, expected);
        assert_eq!(
            config.access_key_id.as_ref().map(SecretValue::expose),
            Some("params-access")
        );
        assert_eq!(config.region_name.as_deref(), Some("eu-central-1"));
    }

    #[rstest::rstest]
    #[case::bedrock(BEDROCK_SERVICE, Some(BEDROCK_SESSION_POLICY))]
    #[case::claude_platform("aws-external-anthropic", Some(BEDROCK_SESSION_POLICY))]
    #[case::textract("textract", None)]
    fn the_session_policy_follows_the_signing_service(
        #[case] service: &str,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(web_identity_session_policy(service), expected);
    }

    #[ignore]
    #[tokio::test]
    async fn live_bedrock_invoke_model_returns_200() -> Result<(), Box<dyn std::error::Error>> {
        let access_key_id = std::env::var("AWS_BEDROCK_TEST_ACCESS_KEY_ID")?;
        let secret_access_key = std::env::var("AWS_BEDROCK_TEST_SECRET_ACCESS_KEY")?;
        let body = br#"{"anthropic_version":"bedrock-2023-05-31","max_tokens":1,"messages":[{"role":"user","content":[{"type":"text","text":"ping"}]}]}"#.to_vec();
        let headers =
            HeaderMap::from_iter([(CONTENT_TYPE, HeaderValue::from_static("application/json"))]);
        let credentials = AwsAuthService::default()
            .resolve_credentials(
                AwsAuthConfig {
                    access_key_id: Some(access_key_id.into()),
                    secret_access_key: Some(secret_access_key.into()),
                    region_name: Some("us-west-2".to_string()),
                    ..Default::default()
                },
                &no_env,
            )
            .await?;
        let client = litellm_http::Client::for_test();
        let mut failures = Vec::new();

        for region in ["us-west-2", "us-east-1"] {
            let url = format!(
                "https://bedrock-runtime.{region}.amazonaws.com/model/us.anthropic.claude-opus-4-8/invoke"
            );
            let signed_headers = sign_headers(
                "POST",
                &url,
                &body,
                &headers,
                SigningScope {
                    region,
                    service: BEDROCK_SERVICE,
                    credentials: &credentials,
                    time: SystemTime::now(),
                },
            )?;
            let response = client
                .post(&url)
                .headers(headers.clone())
                .headers(signed_headers)
                .body(body.clone())
                .send()
                .await?;
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

    #[rstest::rstest]
    #[case::provisioned(
        None,
        "https://cache-1/?Action=connect&User=iam-user",
        "cache-1/?Action=connect&User=iam-user&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIDEXAMPLE%2F20261029%2Fus-east-1%2Felasticache%2Faws4_request&X-Amz-Date=20261029T120000Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host&X-Amz-Signature=dfa58492a6ae9ba065088724170a52437329d03f3e9bc9a4c90a80cc8a18e2f6"
    )]
    #[case::serverless_with_a_session_token(
        Some("session-token"),
        "https://cache-1/?Action=connect&User=iam-user&ResourceType=ServerlessCache",
        "cache-1/?Action=connect&User=iam-user&ResourceType=ServerlessCache&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIDEXAMPLE%2F20261029%2Fus-east-1%2Felasticache%2Faws4_request&X-Amz-Date=20261029T120000Z&X-Amz-Expires=900&X-Amz-SignedHeaders=host&X-Amz-Security-Token=session-token&X-Amz-Signature=e48bdbb3bf9d5e8f1d325bf33d5260e333df1f6b8868e8edd1e4c3cc01b4c3f9"
    )]
    fn presigned_urls_match_botocore_sigv4_query_auth(
        #[case] session_token: Option<&str>,
        #[case] url: &str,
        #[case] expected: &str,
    ) {
        // Expected values from botocore 1.43.117 SigV4QueryAuth with get_current_datetime patched to
        // 2026-10-29T12:00:00Z (generated 2026-10-29), the signer Python's ElastiCache IAM provider uses
        let credentials = Credentials::new(
            "AKIDEXAMPLE",
            "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            session_token.map(str::to_owned),
            None,
            "test",
        );
        let signed = presign_url(
            url,
            Duration::from_secs(900),
            SigningScope {
                region: "us-east-1",
                service: "elasticache",
                credentials: &credentials,
                time: UNIX_EPOCH + Duration::from_secs(1_793_275_200),
            },
        )
        .unwrap();

        assert_eq!(signed.strip_prefix("https://"), Some(expected));
    }
}
