//! google-auth's `external_account.Credentials`: a subject token from the configured source,
//! exchanged at the STS endpoint as `sts.Client.exchange_token` posts it and read as
//! `_perform_refresh_token` reads the reply, then optionally exchanged again for a service
//! account's token through `impersonated_credentials`.

use std::{future::Future, pin::Pin, sync::Arc, time::Duration};

use std::collections::BTreeMap;

use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_python_compat::urllib::quote;
use serde::Deserialize;
use time::{OffsetDateTime, PrimitiveDateTime, macros::format_description};

use super::{
    aws_wif::ExplicitAwsKeys,
    builders::token_failure,
    credentials::{
        Credentials, Expiry, Lifetime, Loaded, Minted, RefreshFuture, TokenReply, TokenState,
        not_a_token_reply,
    },
    external_authorized_user::oauth_error,
    failure::{GoogleLoginFailure, PythonException},
    service_account::invalid_lifetime,
    source::{BoundedRead, MAX_CREDENTIALS_FILE_BYTES, SdkJson, present, read_bounded_off_runtime},
    transport::Requests,
};
use crate::constants::CLOUD_PLATFORM_SCOPE;

pub(crate) type OsEnvironment = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;
pub(crate) type SubjectFuture<'a> =
    Pin<Box<dyn Future<Output = Result<String, GoogleLoginFailure>> + Send + 'a>>;

const STS_GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:token-exchange";
const STS_REQUESTED_TOKEN_TYPE: &str = "urn:ietf:params:oauth:token-type:access_token";
const DEFAULT_TOKEN_URL: &str = "https://sts.{universe_domain}/v1/token";
const DEFAULT_UNIVERSE_DOMAIN: &str = "googleapis.com";
/// `iam._IAM_SCOPE`, which impersonated credentials scope their source credentials to.
const IAM_SCOPE: &str = "https://www.googleapis.com/auth/iam";
const IMPERSONATION_REFUSED: &str = "IAM refused to impersonate the service account";
/// `impersonated_credentials._DEFAULT_TOKEN_LIFETIME_SECS`.
const DEFAULT_IMPERSONATION_LIFETIME: u64 = 3600;
pub(crate) const NO_CREDENTIAL_SOURCE: &str =
    "The external account credentials name no credential_source.";
const WORKFORCE_POOL_PREFIX: &str = "//iam.googleapis.com/locations/";

/// What an external account's `retrieve_subject_token` reads its token from.
pub(crate) trait SubjectToken: Send + Sync {
    fn subject_token<'a>(&'a self, requests: &'a Requests) -> SubjectFuture<'a>;
}

/// An external account's file, as `external_account.Credentials.from_info` and LiteLLM's
/// explicit AWS login read it.
#[derive(Clone, Default, Deserialize, PartialEq, veil::Redact)]
pub(crate) struct ExternalAccountInfo {
    pub(crate) audience: Option<String>,
    pub(crate) subject_token_type: Option<String>,
    pub(crate) token_url: Option<String>,
    pub(crate) service_account_impersonation_url: Option<String>,
    pub(crate) service_account_impersonation: Option<ImpersonationOptions>,
    pub(crate) client_id: Option<String>,
    #[redact(with = "[REDACTED]")]
    pub(crate) client_secret: Option<String>,
    pub(crate) workforce_pool_user_project: Option<String>,
    pub(crate) universe_domain: Option<String>,
    pub(crate) credential_source: Option<CredentialSource>,
    #[serde(flatten)]
    pub(crate) aws: ExplicitAwsKeys,
    #[serde(skip)]
    pub(crate) sdk: SdkJson,
}

/// `service_account_impersonation`'s options.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub(crate) struct ImpersonationOptions {
    pub(crate) token_lifetime_seconds: Option<u64>,
}

/// Where an external account reads its subject token: a file, a URL, a certificate, an
/// executable, or AWS.
#[derive(Clone, Default, Deserialize, PartialEq, veil::Redact)]
pub(crate) struct CredentialSource {
    pub(crate) environment_id: Option<String>,
    pub(crate) executable: Option<ExecutableSource>,
    pub(crate) file: Option<String>,
    pub(crate) url: Option<String>,
    pub(crate) certificate: Option<serde::de::IgnoredAny>,
    #[redact(with = "[REDACTED]")]
    pub(crate) headers: Option<BTreeMap<String, String>>,
    pub(crate) format: Option<SourceFormat>,
    pub(crate) region_url: Option<String>,
    pub(crate) regional_cred_verification_url: Option<String>,
    pub(crate) imdsv2_session_token_url: Option<String>,
}

/// The `format` a file or URL source declares for its content.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub(crate) struct SourceFormat {
    #[serde(rename = "type")]
    pub(crate) kind: Option<String>,
    pub(crate) subject_token_field_name: Option<String>,
}

/// An executable source's command and its limits.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub(crate) struct ExecutableSource {
    pub(crate) command: Option<String>,
    pub(crate) timeout_millis: Option<f64>,
    pub(crate) interactive_timeout_millis: Option<f64>,
    pub(crate) output_file: Option<String>,
}

/// The fields `external_account.Credentials.from_info` reads, as the token exchange uses them.
pub(crate) struct Settings {
    pub(crate) audience: Option<String>,
    pub(crate) subject_token_type: Option<String>,
    pub(crate) token_url: Option<String>,
    pub(crate) impersonation_url: Option<String>,
    pub(crate) impersonation_lifetime: Option<String>,
    pub(crate) client_id: Option<String>,
    pub(crate) client_secret: Option<String>,
    pub(crate) workforce_pool_user_project: Option<String>,
}

/// `is_workforce_pool`: an audience naming a workforce pool.
fn is_workforce_pool(audience: Option<&str>) -> bool {
    audience
        .and_then(|audience| audience.strip_prefix(WORKFORCE_POOL_PREFIX))
        .and_then(|rest| rest.split_once('/'))
        .is_some_and(|(location, rest)| !location.is_empty() && rest.starts_with("workforcePools/"))
}

/// `token_url` with the default STS endpoint's universe domain filled in.
fn token_url(info: &ExternalAccountInfo) -> Option<String> {
    let universe_domain = info
        .universe_domain
        .as_deref()
        .unwrap_or(DEFAULT_UNIVERSE_DOMAIN);
    info.token_url
        .clone()
        .map(|url| match url == DEFAULT_TOKEN_URL {
            true => url.replace("{universe_domain}", universe_domain),
            false => url,
        })
}

impl Settings {
    /// `external_account.Credentials.from_info` and the base `__init__`'s own check.
    pub(crate) fn from_info(info: &ExternalAccountInfo) -> Result<Self, GoogleLoginFailure> {
        let settings = Self {
            audience: info.audience.clone(),
            subject_token_type: info.subject_token_type.clone(),
            token_url: token_url(info),
            impersonation_url: info.service_account_impersonation_url.clone(),
            impersonation_lifetime: info
                .service_account_impersonation
                .as_ref()
                .and_then(|options| options.token_lifetime_seconds)
                .filter(|lifetime| *lifetime != 0)
                .map(|lifetime| lifetime.to_string()),
            client_id: info.client_id.clone(),
            client_secret: info.client_secret.clone(),
            workforce_pool_user_project: info.workforce_pool_user_project.clone(),
        };
        if !is_workforce_pool(settings.audience.as_deref())
            && info
                .workforce_pool_user_project
                .as_ref()
                .is_some_and(|project| !project.is_empty())
        {
            return Err(GoogleLoginFailure::new(
                PythonException::InvalidValue,
                "workforce_pool_user_project is only for workforce pool audiences.",
            ));
        }
        Ok(settings)
    }

    /// `aws.Credentials(...)` as LiteLLM's explicit AWS login builds it: no client
    /// authentication, workforce project or impersonation options.
    pub(crate) fn for_explicit_aws(info: &ExternalAccountInfo) -> Self {
        Self {
            audience: info.audience.clone(),
            subject_token_type: info.subject_token_type.clone(),
            token_url: token_url(info),
            impersonation_url: info.service_account_impersonation_url.clone(),
            impersonation_lifetime: None,
            client_id: None,
            client_secret: None,
            workforce_pool_user_project: None,
        }
    }
}

/// google-auth's external account credentials: a subject token from its source, exchanged at
/// STS, and optionally exchanged again for a service account's token.
pub(crate) struct ExternalAccount {
    settings: Settings,
    subject: Box<dyn SubjectToken>,
    requests: Requests,
    /// The STS token impersonation is asked with, kept until it goes stale.
    source_token: tokio::sync::Mutex<Option<Minted>>,
}

impl ExternalAccount {
    pub(crate) fn loaded(settings: Settings, subject: Box<dyn SubjectToken>) -> Loaded {
        Arc::new(Self::new(settings, subject, Requests::process()))
    }

    pub(crate) fn new(
        settings: Settings,
        subject: Box<dyn SubjectToken>,
        requests: Requests,
    ) -> Self {
        Self {
            settings,
            subject,
            requests,
            source_token: tokio::sync::Mutex::new(None),
        }
    }

    /// `sts.Client.exchange_token`: the form `urlencode` builds from google-auth's dict, empty
    /// fields dropped, with HTTP basic client authentication when a client id is set.
    async fn exchange(&self, scope: &str) -> Result<Minted, GoogleLoginFailure> {
        let subject_token = self.subject.subject_token(&self.requests).await?;
        let options = self
            .settings
            .workforce_pool_user_project
            .as_ref()
            .filter(|project| !project.is_empty() && self.settings.client_id.is_none())
            .map(|project| {
                let options = serde_json::json!({ "userProject": project }).to_string();
                quote(&options.replace("\":\"", "\": \""), "/")
            });
        let fields = [
            ("grant_type", Some(STS_GRANT_TYPE.to_owned())),
            ("audience", self.settings.audience.clone()),
            ("scope", Some(scope.to_owned())),
            (
                "requested_token_type",
                Some(STS_REQUESTED_TOKEN_TYPE.to_owned()),
            ),
            ("subject_token", Some(subject_token)),
            (
                "subject_token_type",
                self.settings.subject_token_type.clone(),
            ),
            ("options", options),
        ];
        let form: Vec<(&str, String)> = fields
            .into_iter()
            .filter_map(|(name, value)| Some((name, value.filter(|value| !value.is_empty())?)))
            .collect();
        let token_url = self.settings.token_url.as_deref().unwrap_or("None");
        let mut request = self
            .requests
            .client(token_url)?
            .post(token_url)
            .header(http::header::ACCEPT, "*/*")
            .form(&form);
        if let Some(client_id) = &self.settings.client_id {
            let credentials = STANDARD.encode(format!(
                "{client_id}:{}",
                self.settings.client_secret.as_deref().unwrap_or_default()
            ));
            request = request.header(http::header::AUTHORIZATION, format!("Basic {credentials}"));
        }
        let response = request
            .send()
            .await
            .map_err(|error| token_failure(&error))?;
        let status = response.status().as_u16();
        let body = response
            .text()
            .await
            .map_err(|error| token_failure(&error))?;
        if status != 200 {
            return Err(oauth_error(&body));
        }
        sts_reply(&body, OffsetDateTime::now_utc())
    }

    /// The STS token impersonation is asked with: refreshed only when it is stale or invalid.
    async fn source_token(&self) -> Result<Minted, GoogleLoginFailure> {
        let mut held = self.source_token.lock().await;
        if let Some(token) = held.as_ref()
            && token.state(OffsetDateTime::now_utc()) == TokenState::Fresh
        {
            return Ok(token.clone());
        }
        let token = self.exchange(IAM_SCOPE).await?;
        *held = Some(token.clone());
        Ok(token)
    }

    /// `impersonated_credentials._make_iam_token_request` with the STS token as the bearer.
    async fn impersonate(&self, url: &str) -> Result<Minted, GoogleLoginFailure> {
        if target_principal(url).is_none() {
            return Err(GoogleLoginFailure::new(
                PythonException::Refresh,
                "The service account impersonation URL names no service account.",
            ));
        }
        let source = self.source_token().await?.usable()?;
        let lifetime = self
            .settings
            .impersonation_lifetime
            .clone()
            .unwrap_or_else(|| DEFAULT_IMPERSONATION_LIFETIME.to_string());
        let body = format!(
            "{{\"delegates\": null, \"scope\": [\"{CLOUD_PLATFORM_SCOPE}\"], \"lifetime\": {}}}",
            serde_json::to_string(&format!("{lifetime}s")).unwrap_or_default()
        );
        let response = self
            .requests
            .client(url)?
            .post(url)
            .header(http::header::ACCEPT, "*/*")
            .header(http::header::CONTENT_TYPE, "application/json")
            .header(
                http::header::AUTHORIZATION,
                format!("Bearer {}", source.expose()),
            )
            .body(body)
            .send()
            .await
            .map_err(|error| token_failure(&error))?;
        let status = response.status().as_u16();
        let body = response
            .text()
            .await
            .map_err(|error| token_failure(&error))?;
        let refused = |message: &str| {
            GoogleLoginFailure::new(PythonException::Refresh, format!("{message}: {body}"))
        };
        if status != 200 {
            return Err(refused(IMPERSONATION_REFUSED));
        }
        // A 200 reply can hold a live access token, so it never goes into the message.
        let malformed = || {
            GoogleLoginFailure::new(
                PythonException::Refresh,
                "IAM's answer has no accessToken or a malformed expireTime",
            )
        };
        let reply = serde_json::from_str::<ImpersonationReply>(&body).map_err(|_| malformed())?;
        let (Some(token), Some(expire_time)) = (reply.access_token, reply.expire_time) else {
            return Err(malformed());
        };
        let expiry = expire_time
            .as_deref()
            .and_then(|text| {
                PrimitiveDateTime::parse(
                    text,
                    format_description!("[year]-[month]-[day]T[hour]:[minute]:[second]Z"),
                )
                .ok()
            })
            .map(PrimitiveDateTime::assume_utc)
            .ok_or_else(malformed)?;
        Ok(Minted::from_reply(token, Expiry::At(expiry)))
    }

    async fn mint(&self) -> Result<Minted, GoogleLoginFailure> {
        match &self.settings.impersonation_url {
            Some(url) => self.impersonate(url).await,
            None => self.exchange(CLOUD_PLATFORM_SCOPE).await,
        }
    }
}

impl Credentials for ExternalAccount {
    fn refresh(&self) -> RefreshFuture<'_> {
        Box::pin(self.mint())
    }
}

/// IAM's `generateAccessToken` reply. Each field is `Some(None)` when null.
#[derive(Deserialize)]
struct ImpersonationReply {
    #[serde(rename = "accessToken", default, deserialize_with = "present")]
    access_token: Option<Option<String>>,
    #[serde(rename = "expireTime", default, deserialize_with = "present")]
    expire_time: Option<Option<String>>,
}

/// `service_account_email`: the text between the last `/` and `:generateAccessToken`.
fn target_principal(url: &str) -> Option<&str> {
    let start = url.rfind('/')?;
    let end = url.find(":generateAccessToken")?;
    (start < end).then(|| &url[start + 1..end])
}

/// `_make_request`'s JSON body and `_perform_refresh_token`'s reading of it: only
/// `access_token` and `expires_in` matter, and either may be missing or of any type.
fn sts_reply(body: &str, now: OffsetDateTime) -> Result<Minted, GoogleLoginFailure> {
    let reply = match body {
        "" => TokenReply::default(),
        body => TokenReply::parse(body).map_err(|_| not_a_token_reply())?,
    };
    let seconds = match &reply.expires_in {
        Some(Lifetime::Text(text)) => text
            .trim()
            .parse::<i64>()
            .map(|seconds| seconds as f64)
            .map_err(|_| invalid_lifetime())?,
        Some(Lifetime::Seconds(seconds)) => *seconds,
        None => return Err(invalid_lifetime()),
    };
    Ok(Minted::from_reply(
        reply.access_token.flatten(),
        Expiry::At(now + time::Duration::seconds_f64(seconds)),
    ))
}

/// How a file or URL source holds its subject token.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Format {
    Text,
    Json(String),
}

/// `identity_pool._parse_token_data`.
fn parse_token_data(
    content: &str,
    location: &str,
    format: &Format,
) -> Result<String, GoogleLoginFailure> {
    let token = match format {
        Format::Text => content.to_owned(),
        Format::Json(field) => {
            let unparseable = || {
                GoogleLoginFailure::new(
                    PythonException::Refresh,
                    format!("The subject token at {location} has no {field} field."),
                )
            };
            let data = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(content)
                .map_err(|_| unparseable())?;
            let token = data.get(field).ok_or_else(unparseable)?;
            token.as_str().unwrap_or_default().to_owned()
        }
    };
    if token.is_empty() {
        return Err(GoogleLoginFailure::new(
            PythonException::Refresh,
            "The subject token source is empty.",
        ));
    }
    Ok(token)
}

/// A subject token read from a file, as `identity_pool` reads one.
struct FileSource {
    path: String,
    format: Format,
}

impl SubjectToken for FileSource {
    fn subject_token<'a>(&'a self, _requests: &'a Requests) -> SubjectFuture<'a> {
        Box::pin(async move {
            let bytes = read_bounded_off_runtime(self.path.clone())
                .await
                .map_err(|failure| match failure {
                    BoundedRead::Unreadable(error)
                        if error.kind() == std::io::ErrorKind::NotFound =>
                    {
                        GoogleLoginFailure::new(
                            PythonException::Refresh,
                            format!("The subject token file {} does not exist.", self.path),
                        )
                    }
                    BoundedRead::Unreadable(error) => {
                        GoogleLoginFailure::new(PythonException::OS, error.to_string())
                    }
                    BoundedRead::TooLarge => too_large_file("subject token file", &self.path),
                })?;
            let content = String::from_utf8(bytes).map_err(|_| {
                GoogleLoginFailure::new(
                    PythonException::OS,
                    format!("The subject token file {} is not UTF-8 text.", self.path),
                )
            })?;
            parse_token_data(&content, &self.path, &self.format)
        })
    }
}

/// A token file over [`MAX_CREDENTIALS_FILE_BYTES`], larger than any token a source can hold.
fn too_large_file(kind: &str, path: &str) -> GoogleLoginFailure {
    GoogleLoginFailure::new(
        PythonException::Refresh,
        format!("The {kind} {path} is over {MAX_CREDENTIALS_FILE_BYTES} bytes."),
    )
}

/// A subject token fetched from a URL, as `identity_pool` fetches one.
struct UrlSource {
    url: String,
    headers: Vec<(String, String)>,
    format: Format,
}

impl SubjectToken for UrlSource {
    fn subject_token<'a>(&'a self, requests: &'a Requests) -> SubjectFuture<'a> {
        Box::pin(async move {
            let request = self.headers.iter().fold(
                requests
                    .client(&self.url)?
                    .get(&self.url)
                    .header(http::header::ACCEPT, "*/*"),
                |request, (name, value)| request.header(name.as_str(), value.as_str()),
            );
            let response = request
                .send()
                .await
                .map_err(|error| token_failure(&error))?;
            let status = response.status().as_u16();
            let body = response
                .text()
                .await
                .map_err(|error| token_failure(&error))?;
            if status != 200 {
                return Err(GoogleLoginFailure::new(
                    PythonException::Refresh,
                    format!("The subject token URL answered with an error: {body}"),
                ));
            }
            parse_token_data(&body, &self.url, &self.format)
        })
    }
}

fn malformed(message: impl Into<String>) -> GoogleLoginFailure {
    GoogleLoginFailure::new(PythonException::Malformed, message)
}

/// `identity_pool._validate_file_or_url_config`.
fn source_format(format: Option<&SourceFormat>) -> Result<Format, GoogleLoginFailure> {
    let kind = format
        .and_then(|format| format.kind.clone())
        .filter(|kind| !kind.is_empty())
        .unwrap_or_else(|| "text".to_owned());
    match kind.as_str() {
        "text" => Ok(Format::Text),
        "json" => match format.and_then(|format| format.subject_token_field_name.clone()) {
            None => Err(malformed(
                "A json credential_source format needs subject_token_field_name.",
            )),
            Some(field) => Ok(Format::Json(field)),
        },
        other => Err(malformed(format!(
            "The credential_source format {other} is neither text nor json."
        ))),
    }
}

/// `identity_pool.Credentials.from_info`. A certificate source, which mints through mutual TLS,
/// is left to Google's Rust SDK, so `Ok(None)` names it.
pub(crate) fn identity_pool(
    info: &ExternalAccountInfo,
) -> Result<Option<Loaded>, GoogleLoginFailure> {
    let settings = Settings::from_info(info)?;
    let Some(source) = &info.credential_source else {
        return Err(GoogleLoginFailure::new(
            PythonException::InvalidValue,
            NO_CREDENTIAL_SOURCE,
        ));
    };
    if source.environment_id.is_some() {
        return Err(malformed(
            "An identity pool credential_source cannot name an environment_id.",
        ));
    }
    let given = [
        source.file.is_some(),
        source.url.is_some(),
        source.certificate.is_some(),
    ];
    match given.iter().filter(|given| **given).count() {
        0 => {
            return Err(malformed(
                "The credential_source names no file, url or certificate.",
            ));
        }
        1 => {}
        _ => {
            return Err(malformed(
                "The credential_source names more than one of file, url and certificate.",
            ));
        }
    }
    if source.certificate.is_some() {
        return Ok(None);
    }
    let format = source_format(source.format.as_ref())?;
    let nonempty = |field: &Option<String>| field.clone().filter(|text| !text.is_empty());
    let subject: Box<dyn SubjectToken> = match (nonempty(&source.file), nonempty(&source.url)) {
        (Some(path), _) => Box::new(FileSource { path, format }),
        (_, Some(url)) => Box::new(UrlSource {
            url,
            headers: source
                .headers
                .clone()
                .unwrap_or_default()
                .into_iter()
                .collect(),
            format,
        }),
        _ => return Ok(None),
    };
    Ok(Some(ExternalAccount::loaded(settings, subject)))
}

/// An executable's JSON answer, as `pluggable._parse_subject_token` reads it.
#[derive(Clone, Default, Deserialize, PartialEq, veil::Redact)]
struct ExecutableResponse {
    version: Option<f64>,
    success: Option<bool>,
    code: Option<String>,
    message: Option<String>,
    expiration_time: Option<f64>,
    token_type: Option<String>,
    #[redact(with = "[REDACTED]")]
    id_token: Option<String>,
    #[redact(with = "[REDACTED]")]
    saml_response: Option<String>,
}

impl ExecutableResponse {
    /// The answer an executable printed or wrote: text that is not JSON is a `ValueError`, and
    /// JSON that is not an executable response a `MalformedError`.
    fn parse(text: &[u8]) -> Result<Self, GoogleLoginFailure> {
        serde_json::from_slice(text).map_err(|error| match error.classify() {
            serde_json::error::Category::Data => malformed_response(),
            _ => GoogleLoginFailure::new(PythonException::Value, error.to_string()),
        })
    }
}

fn malformed_response() -> GoogleLoginFailure {
    malformed("The executable response is not a JSON object.")
}

/// `pluggable._parse_subject_token` over the executable's JSON answer.
fn executable_subject_token(
    response: &ExecutableResponse,
    now: f64,
) -> Result<String, GoogleLoginFailure> {
    let Some(version) = response.version else {
        return Err(malformed("The executable response has no version."));
    };
    if version > 1.0 {
        return Err(GoogleLoginFailure::new(
            PythonException::Refresh,
            format!("The executable answered with version {version}, which is not supported."),
        ));
    }
    let Some(success) = response.success else {
        return Err(malformed("The executable response has no success."));
    };
    if !success {
        let (Some(code), Some(message)) = (&response.code, &response.message) else {
            return Err(malformed(
                "The executable failed without a code and message.",
            ));
        };
        return Err(GoogleLoginFailure::new(
            PythonException::Refresh,
            format!("The executable failed with code {code}: {message}"),
        ));
    }
    if response
        .expiration_time
        .is_some_and(|expiration| expiration < now)
    {
        return Err(GoogleLoginFailure::new(
            PythonException::Refresh,
            "The executable's token has expired.",
        ));
    }
    let Some(token_type) = &response.token_type else {
        return Err(malformed("The executable response has no token_type."));
    };
    let (field, token) = match token_type.as_str() {
        "urn:ietf:params:oauth:token-type:jwt" | "urn:ietf:params:oauth:token-type:id_token" => {
            ("id_token", &response.id_token)
        }
        "urn:ietf:params:oauth:token-type:saml2" => ("saml_response", &response.saml_response),
        _ => {
            return Err(GoogleLoginFailure::new(
                PythonException::Refresh,
                "The executable answered with an unsupported token_type.",
            ));
        }
    };
    token.clone().ok_or_else(|| {
        GoogleLoginFailure::new(
            PythonException::Refresh,
            format!("The executable response has no {field}."),
        )
    })
}

/// A subject token an executable prints, as `pluggable` runs it.
struct Executable {
    command: String,
    timeout: Duration,
    output_file: Option<String>,
    audience: String,
    subject_token_type: String,
    impersonated_email: Option<String>,
    os_env: OsEnvironment,
}

impl Executable {
    /// The token a still-valid answer in `output_file` holds. A file that cannot be read, or is
    /// too large to be an answer, is no cached answer.
    async fn cached(&self) -> Result<Option<String>, GoogleLoginFailure> {
        let Some(path) = &self.output_file else {
            return Ok(None);
        };
        let Ok(content) = read_bounded_off_runtime(path.clone()).await else {
            return Ok(None);
        };
        let response = match ExecutableResponse::parse(&content) {
            Ok(response) => response,
            Err(failure) if failure.exception == PythonException::Malformed => return Err(failure),
            Err(_) => return Ok(None),
        };
        match executable_subject_token(&response, unix_now()) {
            Ok(token) if response.expiration_time.is_some() => Ok(Some(token)),
            Ok(_) => Ok(None),
            Err(failure)
                if matches!(
                    failure.exception,
                    PythonException::Malformed | PythonException::InvalidValue
                ) =>
            {
                Err(failure)
            }
            Err(_) => Ok(None),
        }
    }

    async fn run(&self) -> Result<String, GoogleLoginFailure> {
        if (self.os_env)("GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES").as_deref() != Some("1") {
            return Err(malformed(
                "Set GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES=1 to let credentials run an executable.",
            ));
        }
        if let Some(token) = self.cached().await? {
            return Ok(token);
        }
        let unusable = |problem: &str| {
            GoogleLoginFailure::new(
                PythonException::Value,
                format!("The credential_source executable command {problem}."),
            )
        };
        let words = shlex::split(&self.command).ok_or_else(|| unusable("has an unclosed quote"))?;
        let (program, arguments) = words
            .split_first()
            .ok_or_else(|| unusable("names no program"))?;
        let mut command = tokio::process::Command::new(program);
        command
            .args(arguments)
            .env("GOOGLE_EXTERNAL_ACCOUNT_AUDIENCE", &self.audience)
            .env(
                "GOOGLE_EXTERNAL_ACCOUNT_TOKEN_TYPE",
                &self.subject_token_type,
            )
            .env(
                "GOOGLE_EXTERNAL_ACCOUNT_ID",
                self.impersonated_email.clone().unwrap_or_default(),
            )
            .env("GOOGLE_EXTERNAL_ACCOUNT_INTERACTIVE", "0")
            .env("GOOGLE_EXTERNAL_ACCOUNT_REVOKE", "0")
            .stdin(std::process::Stdio::null())
            .kill_on_drop(true);
        if let Some(email) = &self.impersonated_email {
            command.env("GOOGLE_EXTERNAL_ACCOUNT_IMPERSONATED_EMAIL", email);
        }
        if let Some(path) = &self.output_file {
            command.env("GOOGLE_EXTERNAL_ACCOUNT_OUTPUT_FILE", path);
        }
        let output = tokio::time::timeout(self.timeout, command.output())
            .await
            .map_err(|_| {
                GoogleLoginFailure::new(
                    PythonException::Refresh,
                    format!(
                        "The credential_source executable timed out after {} seconds.",
                        self.timeout.as_secs_f64()
                    ),
                )
            })?
            .map_err(|error| GoogleLoginFailure::new(PythonException::OS, error.to_string()))?;
        let printed = [output.stdout.as_slice(), output.stderr.as_slice()].concat();
        // stdout carries the subject token, so a failed run's output never goes into the
        // message, which can reach the API caller.
        if !output.status.success() {
            return Err(GoogleLoginFailure::new(
                PythonException::Refresh,
                format!(
                    "The credential_source executable exited with code {}.",
                    output.status.code().unwrap_or(-1),
                ),
            ));
        }
        let response = match printed.is_empty() {
            false => ExecutableResponse::parse(&printed)?,
            true => match &self.output_file {
                Some(path) => ExecutableResponse::parse(
                    &read_bounded_off_runtime(path.clone()).await.map_err(
                        |failure| match failure {
                            BoundedRead::Unreadable(error) => {
                                GoogleLoginFailure::new(PythonException::OS, error.to_string())
                            }
                            BoundedRead::TooLarge => too_large_file("executable output file", path),
                        },
                    )?,
                )?,
                None => return Err(malformed_response()),
            },
        };
        executable_subject_token(&response, unix_now())
    }
}

fn unix_now() -> f64 {
    OffsetDateTime::now_utc().unix_timestamp_nanos() as f64 / 1e9
}

impl SubjectToken for Executable {
    fn subject_token<'a>(&'a self, _requests: &'a Requests) -> SubjectFuture<'a> {
        Box::pin(self.run())
    }
}

/// `pluggable.Credentials.from_info`. Whether executables may run is only checked when a token
/// is asked for.
pub(crate) fn pluggable(
    info: &ExternalAccountInfo,
    os_env: OsEnvironment,
) -> Result<Loaded, GoogleLoginFailure> {
    let settings = Settings::from_info(info)?;
    let Some(source) = &info.credential_source else {
        return Err(malformed(
            "The external account credentials name no credential_source.",
        ));
    };
    let Some(executable) = &source.executable else {
        return Err(malformed("The credential_source names no executable."));
    };
    let Some(command) = executable
        .command
        .clone()
        .filter(|command| !command.is_empty())
    else {
        return Err(malformed(
            "The credential_source executable has no command.",
        ));
    };
    let timeout_millis = executable
        .timeout_millis
        .filter(|timeout| *timeout != 0.0)
        .unwrap_or(30_000.0);
    if !(5_000.0..=120_000.0).contains(&timeout_millis) {
        return Err(GoogleLoginFailure::new(
            PythonException::InvalidValue,
            "The executable timeout_millis must be between 5000 and 120000.",
        ));
    }
    if executable
        .interactive_timeout_millis
        .is_some_and(|millis| millis != 0.0 && !(30_000.0..=1_800_000.0).contains(&millis))
    {
        return Err(GoogleLoginFailure::new(
            PythonException::InvalidValue,
            "The executable interactive_timeout_millis must be between 30000 and 1800000.",
        ));
    }
    let impersonated_email = settings
        .impersonation_url
        .as_deref()
        .and_then(target_principal)
        .map(str::to_owned);
    let subject = Executable {
        command,
        timeout: Duration::from_secs_f64(timeout_millis / 1000.0),
        output_file: executable.output_file.clone(),
        audience: settings.audience.clone().unwrap_or_default(),
        subject_token_type: settings.subject_token_type.clone().unwrap_or_default(),
        impersonated_email,
        os_env,
    };
    Ok(ExternalAccount::loaded(settings, Box::new(subject)))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use axum::{Router, routing::post};
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    fn parsed(value: Value) -> ExternalAccountInfo {
        serde_json::from_value(value).unwrap()
    }

    #[rstest]
    fn executable_response_debug_never_shows_a_token() {
        let response = ExecutableResponse::parse(
            br#"{"version": 1, "success": true, "id_token": "credential-value", "saml_response": "credential-value"}"#,
        )
        .unwrap();
        let debug = format!("{response:?}");
        assert!(!debug.contains("credential-value"), "{debug}");
        assert!(debug.contains("[REDACTED]"), "{debug}");
    }

    const NOW: OffsetDateTime = time::macros::datetime!(2026-11-16 12:00 UTC);

    struct Fixed(&'static str);

    impl SubjectToken for Fixed {
        fn subject_token<'a>(&'a self, _requests: &'a Requests) -> SubjectFuture<'a> {
            Box::pin(async { Ok(self.0.to_owned()) })
        }
    }

    /// A request's path, authorization header and body.
    type Seen = (String, Option<String>, String);

    struct Recorded {
        url: String,
        requests: Arc<Mutex<Vec<Seen>>>,
        server: tokio::task::JoinHandle<()>,
    }

    impl Drop for Recorded {
        fn drop(&mut self) {
            self.server.abort();
        }
    }

    /// An STS and IAM endpoint answering each path with its scripted status and body, recording
    /// each request's path, authorization header and body.
    async fn sts(answers: Vec<(&'static str, u16, &'static str)>) -> Recorded {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = requests.clone();
        let answers = Arc::new(answers);
        let app = Router::new().fallback(post(
            move |uri: axum::http::Uri, headers: axum::http::HeaderMap, body: String| {
                let path = uri.path().to_owned();
                recorded.lock().unwrap().push((
                    path.clone(),
                    headers
                        .get(axum::http::header::AUTHORIZATION)
                        .map(|value| value.to_str().unwrap().to_owned()),
                    body,
                ));
                let answers = answers.clone();
                async move {
                    let (_, status, body) = answers
                        .iter()
                        .find(|(answered, _, _)| *answered == path)
                        .copied()
                        .unwrap_or(("", 404, "{}"));
                    (axum::http::StatusCode::from_u16(status).unwrap(), body)
                }
            },
        ));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Recorded {
            url,
            requests,
            server,
        }
    }

    fn account(info: Value, subject: &'static str) -> ExternalAccount {
        ExternalAccount::new(
            Settings::from_info(&parsed(info)).unwrap(),
            Box::new(Fixed(subject)),
            Requests::with_environment(&[]),
        )
    }

    const AUDIENCE: &str =
        "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/q";

    #[tokio::test]
    async fn the_exchange_is_sts_client_s_form_post() {
        let sts = sts(vec![(
            "/token",
            200,
            r#"{"access_token": "sts-token", "expires_in": 3600}"#,
        )])
        .await;
        let account = account(
            json!({
                "audience": AUDIENCE, "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
                "token_url": format!("{}/token", sts.url), "client_id": "id", "client_secret": "secret",
            }),
            "subject",
        );

        assert_eq!(
            account.mint().await.unwrap().token.unwrap().expose(),
            "sts-token"
        );

        let (path, authorization, body) = sts.requests.lock().unwrap()[0].clone();
        assert_eq!(path, "/token");
        assert_eq!(authorization.as_deref(), Some("Basic aWQ6c2VjcmV0"));
        assert_eq!(
            body,
            "grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Atoken-exchange\
             &audience=%2F%2Fiam.googleapis.com%2Fprojects%2F1%2Flocations%2Fglobal%2FworkloadIdentityPools%2Fp%2Fproviders%2Fq\
             &scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcloud-platform\
             &requested_token_type=urn%3Aietf%3Aparams%3Aoauth%3Atoken-type%3Aaccess_token\
             &subject_token=subject\
             &subject_token_type=urn%3Aietf%3Aparams%3Aoauth%3Atoken-type%3Ajwt"
        );
    }

    #[rstest]
    #[case::only_a_token(r#"{"access_token": "t", "expires_in": 60}"#, Some("t"))]
    #[case::string_lifetime(r#"{"access_token": "t", "expires_in": "60"}"#, Some("t"))]
    #[case::no_token(r#"{"token_type": "Bearer", "expires_in": 60}"#, None)]
    fn an_sts_reply_needs_only_what_google_auth_reads(
        #[case] body: &str,
        #[case] token: Option<&str>,
    ) {
        let minted = sts_reply(body, NOW).unwrap();
        assert_eq!(minted.token.as_ref().map(|token| token.expose()), token);
        assert_eq!(minted.expiry, Expiry::At(NOW + time::Duration::seconds(60)));
    }

    #[rstest]
    #[case::no_lifetime(
        r#"{"access_token": "t"}"#,
        "The token endpoint's reply has an expires_in that is not a number of seconds."
    )]
    #[case::bad_lifetime(
        r#"{"access_token": "t", "expires_in": "soon"}"#,
        "The token endpoint's reply has an expires_in that is not a number of seconds."
    )]
    #[case::list(r#"["t"]"#, "The token endpoint's reply is not a JSON object.")]
    #[case::not_json("<html>", "The token endpoint's reply is not a JSON object.")]
    fn an_unreadable_sts_reply_is_a_refresh_error(#[case] body: &str, #[case] message: &str) {
        assert_eq!(
            sts_reply(body, NOW).unwrap_err(),
            GoogleLoginFailure::new(PythonException::Refresh, message)
        );
    }

    #[tokio::test]
    async fn impersonation_asks_iam_with_the_iam_scoped_sts_token() {
        let sts = sts(vec![
            (
                "/token",
                200,
                r#"{"access_token": "sts-token", "expires_in": 3600}"#,
            ),
            (
                "/v1/projects/-/serviceAccounts/sa@p.iam.gserviceaccount.com:generateAccessToken",
                200,
                r#"{"accessToken": "sa-token", "expireTime": "2099-01-01T00:00:00Z"}"#,
            ),
        ])
        .await;
        let account = account(
            json!({
                "audience": AUDIENCE, "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
                "token_url": format!("{}/token", sts.url),
                "service_account_impersonation_url": format!("{}/v1/projects/-/serviceAccounts/sa@p.iam.gserviceaccount.com:generateAccessToken", sts.url),
                "service_account_impersonation": {"token_lifetime_seconds": 600},
            }),
            "subject",
        );

        assert_eq!(
            account.mint().await.unwrap().token.unwrap().expose(),
            "sa-token"
        );
        account.mint().await.unwrap();

        let requests = sts.requests.lock().unwrap().clone();
        assert_eq!(
            requests.len(),
            3,
            "the STS token is reused while it is fresh"
        );
        assert!(
            requests[0]
                .2
                .contains("&scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fiam&")
        );
        assert_eq!(requests[1].1.as_deref(), Some("Bearer sts-token"));
        assert_eq!(
            requests[1].2,
            r#"{"delegates": null, "scope": ["https://www.googleapis.com/auth/cloud-platform"], "lifetime": "600s"}"#
        );
    }

    #[rstest]
    #[case::fractional_expiry(
        r#"{"accessToken": "live-sa-token", "expireTime": "2099-01-01T00:00:00.123Z"}"#
    )]
    #[case::no_expiry(r#"{"accessToken": "live-sa-token"}"#)]
    #[tokio::test]
    async fn a_malformed_iam_answer_keeps_its_token_out_of_the_error(#[case] answer: &'static str) {
        let sts = sts(vec![
            (
                "/token",
                200,
                r#"{"access_token": "sts-token", "expires_in": 3600}"#,
            ),
            (
                "/v1/projects/-/serviceAccounts/sa@p.iam.gserviceaccount.com:generateAccessToken",
                200,
                answer,
            ),
        ])
        .await;
        let account = account(
            json!({
                "audience": AUDIENCE, "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
                "token_url": format!("{}/token", sts.url),
                "service_account_impersonation_url": format!("{}/v1/projects/-/serviceAccounts/sa@p.iam.gserviceaccount.com:generateAccessToken", sts.url),
            }),
            "subject",
        );

        let failure = account.mint().await.unwrap_err();

        assert_eq!(failure.exception, PythonException::Refresh);
        assert_eq!(
            failure.message,
            "IAM's answer has no accessToken or a malformed expireTime"
        );
    }

    #[rstest]
    #[case::missing(
        json!({"file": "/nonexistent/fz-subject.txt"}),
        PythonException::Refresh,
        "The subject token file /nonexistent/fz-subject.txt does not exist."
    )]
    #[case::json_without_field(
        json!({"file": env!("CARGO_MANIFEST_PATH"), "format": {"type": "json", "subject_token_field_name": "t"}}),
        PythonException::Refresh,
        concat!("The subject token at ", env!("CARGO_MANIFEST_PATH"), " has no t field.")
    )]
    #[tokio::test]
    async fn a_file_source_raises_what_identity_pool_raises(
        #[case] source: Value,
        #[case] exception: PythonException,
        #[case] message: &str,
    ) {
        let info = json!({"type": "external_account", "audience": AUDIENCE, "credential_source": source, "token_url": "http://127.0.0.1:9/token"});
        let loaded = identity_pool(&parsed(info)).unwrap().unwrap();
        assert_eq!(
            loaded.refresh().await.unwrap_err(),
            GoogleLoginFailure::new(exception, message)
        );
    }

    /// A text subject token file holding `contents`.
    fn token_file(contents: &[u8]) -> (tempfile::NamedTempFile, FileSource) {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), contents).unwrap();
        let source = FileSource {
            path: file.path().to_str().unwrap().to_owned(),
            format: Format::Text,
        };
        (file, source)
    }

    #[rstest]
    #[case::normal(b"subject-jwt".to_vec(), Ok("subject-jwt"))]
    #[case::oversize(
        vec![b't'; MAX_CREDENTIALS_FILE_BYTES as usize + 1],
        Err(PythonException::Refresh)
    )]
    #[tokio::test]
    async fn a_subject_token_file_is_read_up_to_the_size_limit(
        #[case] contents: Vec<u8>,
        #[case] expected: Result<&str, PythonException>,
    ) {
        let (_file, source) = token_file(&contents);
        let token = source.subject_token(&Requests::with_environment(&[])).await;
        assert_eq!(
            token.as_deref().map_err(|failure| failure.exception),
            expected
        );
    }

    #[rstest]
    #[tokio::test]
    async fn an_oversize_subject_token_file_is_named_in_the_refusal() {
        let (file, source) = token_file(&vec![b't'; MAX_CREDENTIALS_FILE_BYTES as usize + 1]);
        assert_eq!(
            source
                .subject_token(&Requests::with_environment(&[]))
                .await
                .unwrap_err()
                .message,
            format!(
                "The subject token file {} is over 4194304 bytes.",
                file.path().display()
            )
        );
    }

    #[cfg(unix)]
    #[rstest]
    #[tokio::test]
    async fn an_endless_subject_token_file_is_refused_by_its_size() {
        let source = FileSource {
            path: "/dev/zero".into(),
            format: Format::Text,
        };
        assert_eq!(
            source.subject_token(&Requests::with_environment(&[])).await,
            Err(GoogleLoginFailure::new(
                PythonException::Refresh,
                "The subject token file /dev/zero is over 4194304 bytes."
            ))
        );
    }

    #[cfg(unix)]
    #[rstest]
    #[tokio::test]
    async fn an_oversize_executable_output_file_is_refused_by_its_size() {
        let (file, _) = token_file(&vec![b' '; MAX_CREDENTIALS_FILE_BYTES as usize + 1]);
        let path = file.path().to_str().unwrap().to_owned();
        let executable = Executable {
            command: "true".into(),
            timeout: Duration::from_secs(30),
            output_file: Some(path.clone()),
            audience: "aud".into(),
            subject_token_type: "urn:ietf:params:oauth:token-type:jwt".into(),
            impersonated_email: None,
            os_env: Arc::new(|name| {
                (name == "GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES").then(|| "1".into())
            }),
        };
        assert_eq!(
            executable.run().await,
            Err(GoogleLoginFailure::new(
                PythonException::Refresh,
                format!("The executable output file {path} is over 4194304 bytes.")
            ))
        );
    }

    #[rstest]
    #[case::aws(json!({"environment_id": "aws1"}), PythonException::Malformed, "An identity pool credential_source cannot name an environment_id.")]
    #[case::none(json!({}), PythonException::Malformed, "The credential_source names no file, url or certificate.")]
    #[case::two(json!({"file": "/a", "url": "http://b"}), PythonException::Malformed, "The credential_source names more than one of file, url and certificate.")]
    #[case::bad_format(json!({"file": "/a", "format": {"type": "xml"}}), PythonException::Malformed, "The credential_source format xml is neither text nor json.")]
    #[case::json_without_name(json!({"file": "/a", "format": {"type": "json"}}), PythonException::Malformed, "A json credential_source format needs subject_token_field_name.")]
    fn an_identity_pool_source_is_checked_at_load(
        #[case] source: Value,
        #[case] exception: PythonException,
        #[case] message: &str,
    ) {
        let info =
            json!({"type": "external_account", "audience": AUDIENCE, "credential_source": source});
        assert_eq!(
            identity_pool(&parsed(info)).err(),
            Some(GoogleLoginFailure::new(exception, message))
        );
    }

    #[tokio::test]
    async fn executables_run_only_when_allowed() {
        let info = json!({
            "type": "external_account", "audience": AUDIENCE, "token_url": "http://127.0.0.1:9/token",
            "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
            "credential_source": {"executable": {"command": "/nonexistent/fz-exe"}},
        });
        let loaded = pluggable(&parsed(info), Arc::new(|_| None)).unwrap();
        assert_eq!(
            loaded.refresh().await.unwrap_err(),
            GoogleLoginFailure::new(
                PythonException::Malformed,
                "Set GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES=1 to let credentials run an executable."
            )
        );
    }

    #[tokio::test]
    async fn a_failed_executable_s_output_stays_out_of_the_error() {
        let command = r#"sh -c 'printf "%s" "{\"id_token\": \"leaked-jwt\"}"; echo stderr-secret >&2; exit 3'"#;
        let executable = Executable {
            command: command.to_owned(),
            timeout: Duration::from_secs(30),
            output_file: None,
            audience: "aud".into(),
            subject_token_type: "urn:ietf:params:oauth:token-type:jwt".into(),
            impersonated_email: None,
            os_env: Arc::new(|name| {
                (name == "GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES").then(|| "1".into())
            }),
        };

        let failure = executable
            .subject_token(&Requests::with_environment(&[]))
            .await
            .unwrap_err();

        assert_eq!(failure.exception, PythonException::Refresh);
        assert_eq!(
            failure.message,
            "The credential_source executable exited with code 3."
        );
    }

    #[tokio::test]
    async fn an_allowed_executable_s_token_is_exchanged() {
        let sts = sts(vec![(
            "/token",
            200,
            r#"{"access_token": "exe-token", "expires_in": 3600}"#,
        )])
        .await;
        let command = r#"sh -c 'printf "%s" "{\"version\": 1, \"success\": true, \"token_type\": \"urn:ietf:params:oauth:token-type:jwt\", \"id_token\": \"$GOOGLE_EXTERNAL_ACCOUNT_AUDIENCE-jwt\"}"'"#;
        let info = json!({
            "type": "external_account", "audience": "aud", "token_url": format!("{}/token", sts.url),
            "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
            "credential_source": {"executable": {"command": command}},
        });
        let account = ExternalAccount::new(
            Settings::from_info(&parsed(info)).unwrap(),
            Box::new(Executable {
                command: command.to_owned(),
                timeout: Duration::from_secs(30),
                output_file: None,
                audience: "aud".into(),
                subject_token_type: "urn:ietf:params:oauth:token-type:jwt".into(),
                impersonated_email: None,
                os_env: Arc::new(|name| {
                    (name == "GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES").then(|| "1".into())
                }),
            }),
            Requests::with_environment(&[]),
        );

        assert_eq!(
            account.mint().await.unwrap().token.unwrap().expose(),
            "exe-token"
        );
        assert!(
            sts.requests.lock().unwrap()[0]
                .2
                .contains("&subject_token=aud-jwt&")
        );
    }

    #[rstest]
    #[case::timeout(json!({"command": "x", "timeout_millis": 1000}), "The executable timeout_millis must be between 5000 and 120000.")]
    #[case::interactive(json!({"command": "x", "interactive_timeout_millis": 1000}), "The executable interactive_timeout_millis must be between 30000 and 1800000.")]
    fn executable_timeouts_are_bounded(#[case] executable: Value, #[case] message: &str) {
        let info = json!({"type": "external_account", "audience": AUDIENCE, "credential_source": {"executable": executable}});
        assert_eq!(
            pluggable(&parsed(info), Arc::new(|_| None)).err(),
            Some(GoogleLoginFailure::new(
                PythonException::InvalidValue,
                message
            ))
        );
    }

    #[rstest]
    #[case::workload(AUDIENCE, false)]
    #[case::workforce(
        "//iam.googleapis.com/locations/global/workforcePools/p/providers/q",
        true
    )]
    fn a_workforce_project_needs_a_workforce_audience(
        #[case] audience: &str,
        #[case] accepted: bool,
    ) {
        let info = json!({"audience": audience, "workforce_pool_user_project": "123"});
        assert_eq!(Settings::from_info(&parsed(info)).is_ok(), accepted);
    }

    #[rstest]
    #[case::canonical(
        "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/sa@p.iam.gserviceaccount.com:generateAccessToken",
        Some("sa@p.iam.gserviceaccount.com")
    )]
    #[case::other_host("https://example.com/v1/sa:generateAccessToken", Some("sa"))]
    #[case::no_verb("https://example.com/v1/sa", None)]
    fn the_target_principal_is_read_as_service_account_email_reads_it(
        #[case] url: &str,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(target_principal(url), expected);
    }
}
