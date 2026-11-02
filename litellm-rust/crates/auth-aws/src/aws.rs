use std::time::Duration;
use std::time::{SystemTime, UNIX_EPOCH};

use aws_credential_types::provider::{SharedCredentialsProvider, error::CredentialsError};
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

use litellm_auth_types::AwsParams;

use super::constants::{
    AWS_ACCESS_KEY_ID, AWS_EXTERNAL_ID, AWS_PROFILE_NAME, AWS_REGION_NAME, AWS_ROLE_ARN,
    AWS_ROLE_NAME, AWS_SECRET_ACCESS_KEY, AWS_SESSION_NAME, AWS_SESSION_TOKEN,
    AWS_SIGNED_HEADER_NAMES, AWS_STS_ENDPOINT, AWS_WEB_IDENTITY_TOKEN, AWS_WEB_IDENTITY_TOKEN_FILE,
    DEFAULT_BEDROCK_REGION, DEFAULT_SESSION_NAME_PREFIX, SIGV4_COMPUTED_HEADER_NAMES,
};
use crate::error::{CredentialFlow, Error, SigningStage};

const CREDENTIAL_CACHE_TTL: Duration = Duration::from_secs(600);
const EXPIRY_BUFFER: Duration = Duration::from_secs(60);

#[derive(Clone)]
pub struct AwsAuthService {
    providers: Cache<AwsAuthConfig, SharedCredentialsProvider>,
    credentials: Cache<AwsAuthConfig, Credentials>,
    loader: Arc<dyn ProviderLoader>,
}

impl Default for AwsAuthService {
    fn default() -> Self {
        Self::new(Arc::new(NativeProviderLoader))
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

struct NativeProviderLoader;

impl ProviderLoader for NativeProviderLoader {
    fn load(&self, config: AwsAuthConfig, flow: AwsAuthFlow) -> ProviderFuture<'_> {
        Box::pin(build_provider(config, flow))
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
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AwsAuthFlow {
    WebIdentity {
        token: SecretValue,
        role: String,
        session_name: String,
    },
    AssumeRole {
        role: String,
        session_name: Option<String>,
        web_identity_role: Option<String>,
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
            web_identity_role,
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
        let cached = self.cached_credentials(&resolved, &flow).await?;
        if fresh(&cached) {
            return Ok(cached);
        }
        self.credentials.invalidate(&resolved).await;
        let refetched = self.cached_credentials(&resolved, &flow).await?;
        if fresh(&refetched) {
            return Ok(refetched);
        }
        self.credentials.invalidate(&resolved).await;
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
        } => {
            if is_already_running_as_role(&role, web_identity_role.as_deref(), &resolved).await? {
                return Ok(SharedCredentialsProvider::new(
                    aws_config::default_provider::credentials::DefaultCredentialsChain::builder()
                        .build()
                        .await,
                ));
            }
            let sdk_config = sdk_config(resolved.clone()).await;
            let builder = aws_config::sts::AssumeRoleProvider::builder(role)
                .session_name(session_name.unwrap_or_else(default_session_name));
            let builder = match resolved.external_id {
                Some(id) => builder.external_id(id.expose()),
                None => builder,
            };
            Ok(SharedCredentialsProvider::new(
                builder.configure(&sdk_config).build().await,
            ))
        }
        AwsAuthFlow::WebIdentity {
            token,
            role,
            session_name,
        } => Ok(SharedCredentialsProvider::new(WebIdentityProvider {
            client: aws_sdk_sts::Client::new(&sdk_config(resolved).await),
            token,
            role,
            session_name,
        })),
    }
}

async fn sdk_config(config: AwsAuthConfig) -> aws_types::SdkConfig {
    let loader = aws_config::defaults(aws_config::BehaviorVersion::latest());
    let loader = match config.region_name {
        Some(region) => loader.region(aws_types::region::Region::new(region)),
        None => loader,
    };
    let loader = match config.sts_endpoint {
        Some(endpoint) => loader.endpoint_url(endpoint),
        None => loader,
    };
    let loader = match (config.access_key_id, config.secret_access_key) {
        (Some(key), Some(secret)) => loader.credentials_provider(Credentials::new(
            key.expose(),
            secret.expose(),
            config.session_token.map(|token| token.expose().to_owned()),
            None,
            "litellm-role-source",
        )),
        _ => loader,
    };
    loader.load().await
}

#[derive(Debug)]
struct WebIdentityProvider {
    client: aws_sdk_sts::Client,
    token: SecretValue,
    role: String,
    session_name: String,
}

impl ProvideCredentials for WebIdentityProvider {
    fn provide_credentials<'a>(
        &'a self,
    ) -> aws_credential_types::provider::future::ProvideCredentials<'a>
    where
        Self: 'a,
    {
        aws_credential_types::provider::future::ProvideCredentials::new(async {
            let response = self
                .client
                .assume_role_with_web_identity()
                .role_arn(&self.role)
                .role_session_name(&self.session_name)
                .web_identity_token(self.token.expose())
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

async fn is_already_running_as_role(
    role: &str,
    web_identity_role: Option<&str>,
    config: &AwsAuthConfig,
) -> Result<bool, Error> {
    if config.access_key_id.is_some() || config.secret_access_key.is_some() {
        return Ok(false);
    }
    if role_identity(role).is_none() {
        return Ok(false);
    }
    if let Some(current_role) = web_identity_role {
        return Ok(same_role_arns(role, current_role));
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
    resolve_aws_region(model_region, params, env_lookup)
        .unwrap_or_else(|| DEFAULT_BEDROCK_REGION.to_string())
}

pub fn aws_auth_config(
    params: &AwsParams,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> AwsAuthConfig {
    AwsAuthConfig {
        access_key_id: params
            .resolve(&AwsParams::ACCESS_KEY_ID, env_lookup)
            .map(SecretValue::new),
        secret_access_key: params
            .resolve(&AwsParams::SECRET_ACCESS_KEY, env_lookup)
            .map(SecretValue::new),
        session_token: params
            .resolve(&AwsParams::SESSION_TOKEN, env_lookup)
            .map(SecretValue::new),
        region_name: params.resolve(&AwsParams::REGION, env_lookup),
        session_name: params.resolve(&AwsParams::SESSION_NAME, env_lookup),
        profile_name: params.resolve(&AwsParams::PROFILE_NAME, env_lookup),
        role_name: params.resolve(&AwsParams::ROLE_NAME, env_lookup),
        web_identity_token: params
            .resolve(&AwsParams::WEB_IDENTITY_TOKEN, env_lookup)
            .map(SecretValue::new),
        sts_endpoint: params.resolve(&AwsParams::STS_ENDPOINT, env_lookup),
        external_id: params
            .resolve(&AwsParams::EXTERNAL_ID, env_lookup)
            .map(SecretValue::new),
    }
}

/// Where the credentials that sign a request come from, decided when the request is
/// prepared and resolved when it is sent.
#[derive(Clone, Debug, PartialEq)]
pub enum AwsCredentialSource {
    HostSupplied(Credentials),
    Chain(AwsAuthConfig),
}

impl AwsCredentialSource {
    pub fn from_params(params: &AwsParams, env_lookup: &dyn Fn(&str) -> Option<String>) -> Self {
        match host_supplied_credentials(params) {
            Some(credentials) => Self::HostSupplied(credentials),
            None => Self::Chain(aws_auth_config(params, env_lookup)),
        }
    }

    pub async fn resolve(
        self,
        auth: &AwsAuthService,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Credentials, Error> {
        match self {
            Self::HostSupplied(credentials) => Ok(credentials),
            Self::Chain(config) => auth.resolve_credentials(config, env_lookup).await,
        }
    }
}

/// Credentials a host resolved through its own chain and handed down verbatim.
///
/// A host with its own resolution (LiteLLM's Python `BaseAWSLLM`, which reads
/// profiles, STS and boto sessions) passes the result here so the core signs
/// with exactly those. Without this the core would re-derive from ambient
/// state, where an unrelated `AWS_ROLE_NAME` or `AWS_PROFILE_NAME` in the
/// environment outranks explicit keys in [`classify_auth`] and the two sides
/// would sign as different principals.
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
        let config = AwsAuthConfig {
            access_key_id: access_key.map(SecretValue::new),
            secret_access_key: secret_key.map(SecretValue::new),
            ..Default::default()
        };
        assert!(
            !is_already_running_as_role(role, Some(role), &config)
                .await
                .unwrap()
        );
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
        resolve_aws_region(None, &AwsParams::default(), &env);
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
    #[case::blank_param_falls_to_environment(Some("  "), &[(AWS_SECRET_ACCESS_KEY, "from-env")], Some("from-env"))]
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
    #[case::aws_region(None, None, &[(AWS_REGION, "sa-east-1")], Some("sa-east-1"))]
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
            resolve_bedrock_region(model_region, &params, &env),
            expected.unwrap_or(DEFAULT_BEDROCK_REGION)
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
    fn assume_role_reads_the_ambient_role_from_the_injected_environment(
        #[case] token_file: &str,
        #[case] expected: Option<&str>,
    ) {
        let env = |key: &str| match key {
            AWS_WEB_IDENTITY_TOKEN_FILE => Some(token_file.to_string()),
            AWS_ROLE_ARN => Some("arn:aws:iam::1:role/current".to_string()),
            _ => None,
        };
        let config = AwsAuthConfig {
            role_name: Some("arn:aws:iam::1:role/target".into()),
            ..Default::default()
        };
        let AwsAuthFlow::AssumeRole {
            web_identity_role, ..
        } = classify_auth(config, &env)
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
    #[case::web_identity(AwsAuthFlow::WebIdentity { token: "credential-value".into(), role: "role".into(), session_name: "session".into() })]
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
