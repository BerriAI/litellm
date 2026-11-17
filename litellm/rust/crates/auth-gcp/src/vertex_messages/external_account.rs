//! google-auth's `external_account.Credentials`: a subject token from the configured source,
//! exchanged at the STS endpoint as `sts.Client.exchange_token` posts it and read as
//! `_perform_refresh_token` reads the reply, then optionally exchanged again for a service
//! account's token through `impersonated_credentials`.

use std::{future::Future, pin::Pin, sync::Arc, time::Duration};

use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_python_compat::truthy::json_truthy;
use serde_json::{Map, Value};
use time::{OffsetDateTime, PrimitiveDateTime, macros::format_description};

use super::{
    builders::token_failure,
    credentials::{Credentials, Expiry, Loaded, Minted, RefreshFuture, TokenState},
    external_authorized_user::oauth_error,
    failure::{GoogleLoginFailure, PythonException, python_str},
    service_account::{invalid_lifetime, not_an_object},
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
const IMPERSONATION_REFUSED: &str = "Unable to acquire impersonated credentials";
/// `impersonated_credentials._DEFAULT_TOKEN_LIFETIME_SECS`.
const DEFAULT_IMPERSONATION_LIFETIME: u64 = 3600;
pub(crate) const NO_CREDENTIAL_SOURCE: &str =
    "A valid credential source or a subject token supplier must be provided.";
const WORKFORCE_POOL_PREFIX: &str = "//iam.googleapis.com/locations/";

/// What an external account's `retrieve_subject_token` reads its token from.
pub(crate) trait SubjectToken: Send + Sync {
    fn subject_token<'a>(&'a self, requests: &'a Requests) -> SubjectFuture<'a>;
}

/// The fields `external_account.Credentials.from_info` reads, as Python holds them.
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

/// Python's `str()` of a field that is not null, which is what `urlencode` sends for it.
pub(crate) fn text(info: &Map<String, Value>, field: &str) -> Option<String> {
    info.get(field)
        .filter(|value| !value.is_null())
        .map(|value| python_str(Some(value)))
}

/// `is_workforce_pool`: an audience naming a workforce pool.
fn is_workforce_pool(audience: Option<&str>) -> bool {
    audience
        .and_then(|audience| audience.strip_prefix(WORKFORCE_POOL_PREFIX))
        .and_then(|rest| rest.split_once('/'))
        .is_some_and(|(location, rest)| !location.is_empty() && rest.starts_with("workforcePools/"))
}

impl Settings {
    /// `external_account.Credentials.from_info` and the base `__init__`'s own check.
    pub(crate) fn from_info(info: &Map<String, Value>) -> Result<Self, GoogleLoginFailure> {
        let universe_domain = match info.get("universe_domain") {
            None => DEFAULT_UNIVERSE_DOMAIN.to_owned(),
            Some(domain) => python_str(Some(domain)),
        };
        let settings = Self {
            audience: text(info, "audience"),
            subject_token_type: text(info, "subject_token_type"),
            token_url: text(info, "token_url").map(|url| match url == DEFAULT_TOKEN_URL {
                true => url.replace("{universe_domain}", &universe_domain),
                false => url,
            }),
            impersonation_url: text(info, "service_account_impersonation_url"),
            impersonation_lifetime: info
                .get("service_account_impersonation")
                .filter(|options| json_truthy(Some(options)))
                .and_then(|options| options.get("token_lifetime_seconds"))
                .filter(|lifetime| json_truthy(Some(lifetime)))
                .map(|lifetime| python_str(Some(lifetime))),
            client_id: text(info, "client_id"),
            client_secret: text(info, "client_secret"),
            workforce_pool_user_project: text(info, "workforce_pool_user_project"),
        };
        if !is_workforce_pool(settings.audience.as_deref())
            && json_truthy(info.get("workforce_pool_user_project"))
        {
            return Err(GoogleLoginFailure::new(
                PythonException::InvalidValue,
                "workforce_pool_user_project should not be set for non-workforce pool credentials",
            ));
        }
        Ok(settings)
    }

    /// `aws.Credentials(...)` as LiteLLM's explicit AWS login builds it: no client
    /// authentication, workforce project or impersonation options.
    pub(crate) fn for_explicit_aws(info: &Map<String, Value>) -> Self {
        let universe_domain =
            text(info, "universe_domain").unwrap_or_else(|| DEFAULT_UNIVERSE_DOMAIN.to_owned());
        Self {
            audience: text(info, "audience"),
            subject_token_type: text(info, "subject_token_type"),
            token_url: text(info, "token_url").map(|url| match url == DEFAULT_TOKEN_URL {
                true => url.replace("{universe_domain}", &universe_domain),
                false => url,
            }),
            impersonation_url: text(info, "service_account_impersonation_url"),
            impersonation_lifetime: None,
            client_id: None,
            client_secret: None,
            workforce_pool_user_project: None,
        }
    }
}

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
                percent_encoding::utf8_percent_encode(
                    &options.replace("\":\"", "\": \""),
                    PYTHON_QUOTE_SAFE,
                )
                .to_string()
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
                "Unable to determine target principal from service account impersonation URL.",
            ));
        }
        let source = self.source_token().await?;
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
                format!("Bearer {}", source.value.expose()),
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
        let malformed = || {
            refused(&format!(
                "{IMPERSONATION_REFUSED}: No access token or invalid expiration in response."
            ))
        };
        let data = serde_json::from_str::<Value>(&body).map_err(|_| malformed())?;
        let (Some(token), Some(expire_time)) = (data.get("accessToken"), data.get("expireTime"))
        else {
            return Err(malformed());
        };
        let expiry = expire_time
            .as_str()
            .and_then(|text| {
                PrimitiveDateTime::parse(
                    text,
                    format_description!("[year]-[month]-[day]T[hour]:[minute]:[second]Z"),
                )
                .ok()
            })
            .map(PrimitiveDateTime::assume_utc)
            .ok_or_else(malformed)?;
        Ok(Minted::from_reply(Some(token), Expiry::At(expiry)))
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

/// `urllib.parse.quote`'s safe characters.
const PYTHON_QUOTE_SAFE: &percent_encoding::AsciiSet = &percent_encoding::NON_ALPHANUMERIC
    .remove(b'_')
    .remove(b'.')
    .remove(b'-')
    .remove(b'~')
    .remove(b'/');

/// `service_account_email`: the text between the last `/` and `:generateAccessToken`.
fn target_principal(url: &str) -> Option<&str> {
    let start = url.rfind('/')?;
    let end = url.find(":generateAccessToken")?;
    (start < end).then(|| &url[start + 1..end])
}

/// `_make_request`'s JSON body and `_perform_refresh_token`'s reading of it: only
/// `access_token` and `expires_in` matter, and either may be missing or of any type.
fn sts_reply(body: &str, now: OffsetDateTime) -> Result<Minted, GoogleLoginFailure> {
    let data = match body {
        "" => Map::new(),
        body => match serde_json::from_str::<Value>(body) {
            Ok(Value::Object(data)) => data,
            _ => return Err(not_an_object()),
        },
    };
    let seconds = match data.get("expires_in") {
        Some(Value::String(text)) => text
            .trim()
            .parse::<i64>()
            .map(|seconds| seconds as f64)
            .map_err(|_| invalid_lifetime())?,
        Some(Value::Number(number)) => number.as_f64().unwrap_or_default(),
        _ => return Err(invalid_lifetime()),
    };
    Ok(Minted::from_reply(
        data.get("access_token"),
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
        Format::Text => Value::String(content.to_owned()),
        Format::Json(field) => {
            let unparseable = || {
                GoogleLoginFailure::new(
                    PythonException::Refresh,
                    format!(
                        "Unable to parse subject_token from JSON file '{location}' using key '{field}'"
                    ),
                )
            };
            let Ok(Value::Object(data)) = serde_json::from_str::<Value>(content) else {
                return Err(unparseable());
            };
            data.get(field).cloned().ok_or_else(unparseable)?
        }
    };
    if !json_truthy(Some(&token)) {
        return Err(GoogleLoginFailure::new(
            PythonException::Refresh,
            "Missing subject_token in the credential_source file",
        ));
    }
    Ok(python_str(Some(&token)))
}

struct FileSource {
    path: String,
    format: Format,
}

impl SubjectToken for FileSource {
    fn subject_token<'a>(&'a self, _requests: &'a Requests) -> SubjectFuture<'a> {
        Box::pin(async move {
            if !std::path::Path::new(&self.path).exists() {
                return Err(GoogleLoginFailure::new(
                    PythonException::Refresh,
                    format!("File '{}' was not found.", self.path),
                ));
            }
            let content = std::fs::read_to_string(&self.path)
                .map_err(|error| GoogleLoginFailure::new(PythonException::OS, error.to_string()))?;
            parse_token_data(&content, &self.path, &self.format)
        })
    }
}

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
                    format!("Unable to retrieve Identity Pool subject token: {body}"),
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
fn source_format(source: &Map<String, Value>) -> Result<Format, GoogleLoginFailure> {
    let empty = Value::Object(Map::new());
    let format = source.get("format").unwrap_or(&empty);
    let Value::Object(format) = format else {
        return Err(malformed(
            "Invalid credential_source format. The format is not a dict.",
        ));
    };
    let kind = format
        .get("type")
        .filter(|kind| json_truthy(Some(kind)))
        .map_or_else(|| "text".to_owned(), |kind| python_str(Some(kind)));
    match kind.as_str() {
        "text" => Ok(Format::Text),
        "json" => match format.get("subject_token_field_name") {
            None | Some(Value::Null) => Err(malformed(
                "Missing subject_token_field_name for JSON credential_source format",
            )),
            Some(field) => Ok(Format::Json(python_str(Some(field)))),
        },
        other => Err(malformed(format!(
            "Invalid credential_source format '{other}'"
        ))),
    }
}

/// `identity_pool.Credentials.from_info`. A certificate source, which mints through mutual TLS,
/// is left to Google's Rust SDK, so `Ok(None)` names it.
pub(crate) fn identity_pool(
    info: &Map<String, Value>,
) -> Result<Option<Loaded>, GoogleLoginFailure> {
    let settings = Settings::from_info(info)?;
    let source = match info.get("credential_source") {
        None | Some(Value::Null) => {
            return Err(GoogleLoginFailure::new(
                PythonException::InvalidValue,
                NO_CREDENTIAL_SOURCE,
            ));
        }
        Some(Value::Object(source)) => source,
        Some(_) => {
            return Err(malformed(
                "Invalid credential_source. The credential_source is not a dict.",
            ));
        }
    };
    if source.contains_key("environment_id") {
        return Err(malformed(
            "Invalid Identity Pool credential_source field 'environment_id'",
        ));
    }
    let present = |name: &str| source.get(name).filter(|value| !value.is_null());
    let (file, url, certificate) = (present("file"), present("url"), present("certificate"));
    match [file, url, certificate].iter().flatten().count() {
        0 => {
            return Err(malformed(
                "Missing credential_source. A 'file', 'url', or 'certificate' must be provided.",
            ));
        }
        1 => {}
        _ => {
            return Err(malformed(
                "Ambiguous credential_source. 'file', 'url', and 'certificate' are mutually exclusive..",
            ));
        }
    }
    if certificate.is_some() {
        return Ok(None);
    }
    let format = source_format(source)?;
    let subject: Box<dyn SubjectToken> = match (file, url) {
        (Some(path), _) if json_truthy(Some(path)) => Box::new(FileSource {
            path: python_str(Some(path)),
            format,
        }),
        (_, Some(url)) if json_truthy(Some(url)) => Box::new(UrlSource {
            url: python_str(Some(url)),
            headers: match source.get("headers") {
                Some(Value::Object(headers)) => headers
                    .iter()
                    .map(|(name, value)| (name.clone(), python_str(Some(value))))
                    .collect(),
                _ => Vec::new(),
            },
            format,
        }),
        _ => return Ok(None),
    };
    Ok(Some(ExternalAccount::loaded(settings, subject)))
}

/// `pluggable._parse_subject_token` over the executable's JSON answer.
fn executable_subject_token(response: &Value, now: f64) -> Result<String, GoogleLoginFailure> {
    let Value::Object(response) = response else {
        return Err(malformed("The executable response is not a JSON object."));
    };
    let Some(version) = response.get("version") else {
        return Err(malformed(
            "The executable response is missing the version field.",
        ));
    };
    if version.as_f64().is_some_and(|version| version > 1.0) {
        return Err(GoogleLoginFailure::new(
            PythonException::Refresh,
            format!(
                "Executable returned unsupported version {}.",
                python_str(Some(version))
            ),
        ));
    }
    let Some(success) = response.get("success") else {
        return Err(malformed(
            "The executable response is missing the success field.",
        ));
    };
    if !json_truthy(Some(success)) {
        let (Some(code), Some(message)) = (response.get("code"), response.get("message")) else {
            return Err(malformed(
                "Error code and message fields are required in the response.",
            ));
        };
        return Err(GoogleLoginFailure::new(
            PythonException::Refresh,
            format!(
                "Executable returned unsuccessful response: code: {}, message: {}.",
                python_str(Some(code)),
                python_str(Some(message))
            ),
        ));
    }
    if response
        .get("expiration_time")
        .and_then(Value::as_f64)
        .is_some_and(|expiration| expiration < now)
    {
        return Err(GoogleLoginFailure::new(
            PythonException::Refresh,
            "The token returned by the executable is expired.",
        ));
    }
    let Some(token_type) = response.get("token_type") else {
        return Err(malformed(
            "The executable response is missing the token_type field.",
        ));
    };
    let field = match token_type.as_str() {
        Some(
            "urn:ietf:params:oauth:token-type:jwt" | "urn:ietf:params:oauth:token-type:id_token",
        ) => "id_token",
        Some("urn:ietf:params:oauth:token-type:saml2") => "saml_response",
        _ => {
            return Err(GoogleLoginFailure::new(
                PythonException::Refresh,
                "Executable returned unsupported token type.",
            ));
        }
    };
    response
        .get(field)
        .map(|token| python_str(Some(token)))
        .ok_or_else(|| GoogleLoginFailure::new(PythonException::Refresh, format!("'{field}'")))
}

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
    fn cached(&self) -> Result<Option<String>, GoogleLoginFailure> {
        let Some(path) = &self.output_file else {
            return Ok(None);
        };
        let Some(response) = std::fs::read_to_string(path)
            .ok()
            .and_then(|content| serde_json::from_str::<Value>(&content).ok())
        else {
            return Ok(None);
        };
        match executable_subject_token(&response, unix_now()) {
            Ok(token) if response.get("expiration_time").is_some() => Ok(Some(token)),
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
                "Executables need to be explicitly allowed (set GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES to '1') to run.",
            ));
        }
        if let Some(token) = self.cached()? {
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
                        "Command '{words:?}' timed out after {} seconds",
                        self.timeout.as_secs_f64()
                    ),
                )
            })?
            .map_err(|error| GoogleLoginFailure::new(PythonException::OS, error.to_string()))?;
        let printed = [output.stdout.as_slice(), output.stderr.as_slice()].concat();
        if !output.status.success() {
            return Err(GoogleLoginFailure::new(
                PythonException::Refresh,
                format!(
                    "Executable exited with non-zero return code {}. Error: {}",
                    output.status.code().unwrap_or(-1),
                    String::from_utf8_lossy(&printed).trim_end()
                ),
            ));
        }
        let response = match printed.is_empty() {
            false => serde_json::from_slice::<Value>(&printed).map_err(|error| {
                GoogleLoginFailure::new(PythonException::Value, error.to_string())
            })?,
            true => match &self.output_file {
                Some(path) => {
                    serde_json::from_str::<Value>(&std::fs::read_to_string(path).map_err(
                        |error| GoogleLoginFailure::new(PythonException::OS, error.to_string()),
                    )?)
                    .map_err(|error| {
                        GoogleLoginFailure::new(PythonException::Value, error.to_string())
                    })?
                }
                None => Value::Null,
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
    info: &Map<String, Value>,
    os_env: OsEnvironment,
) -> Result<Loaded, GoogleLoginFailure> {
    let settings = Settings::from_info(info)?;
    let source = match info.get("credential_source") {
        Some(Value::Object(source)) => source,
        _ => {
            return Err(malformed(
                "Missing credential_source. The credential_source is not a dict.",
            ));
        }
    };
    let executable = source
        .get("executable")
        .filter(|executable| json_truthy(Some(executable)));
    let Some(executable) = executable else {
        return Err(malformed(
            "Missing credential_source. An 'executable' must be provided.",
        ));
    };
    let Value::Object(executable) = executable else {
        return Err(malformed(
            "Missing credential_source. The executable is not a dict.",
        ));
    };
    let Some(command) = executable
        .get("command")
        .filter(|command| json_truthy(Some(command)))
    else {
        return Err(malformed(
            "Missing command field. Executable command must be provided.",
        ));
    };
    let timeout_millis = match executable.get("timeout_millis") {
        timeout if !json_truthy(timeout) => 30_000.0,
        Some(timeout) => timeout.as_f64().unwrap_or(f64::NAN),
        None => 30_000.0,
    };
    if !(5_000.0..=120_000.0).contains(&timeout_millis) {
        return Err(GoogleLoginFailure::new(
            PythonException::InvalidValue,
            "Timeout must be between 5 and 120 seconds.",
        ));
    }
    if let Some(interactive) = executable
        .get("interactive_timeout_millis")
        .filter(|timeout| json_truthy(Some(timeout)))
        && !interactive
            .as_f64()
            .is_some_and(|millis| (30_000.0..=1_800_000.0).contains(&millis))
    {
        return Err(GoogleLoginFailure::new(
            PythonException::InvalidValue,
            "Interactive timeout must be between 30 seconds and 30 minutes.",
        ));
    }
    let impersonated_email = settings
        .impersonation_url
        .as_deref()
        .and_then(target_principal)
        .map(str::to_owned);
    let subject = Executable {
        command: python_str(Some(command)),
        timeout: Duration::from_secs_f64(timeout_millis / 1000.0),
        output_file: executable
            .get("output_file")
            .filter(|path| !path.is_null())
            .map(|path| python_str(Some(path))),
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
    use serde_json::json;

    use super::*;

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
            Settings::from_info(info.as_object().unwrap()).unwrap(),
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

        assert_eq!(account.mint().await.unwrap().value.expose(), "sts-token");

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
    #[case::only_a_token(r#"{"access_token": "t", "expires_in": 60}"#, Some("t"), None)]
    #[case::string_lifetime(r#"{"access_token": "t", "expires_in": "60"}"#, Some("t"), None)]
    #[case::no_token(
        r#"{"token_type": "Bearer", "expires_in": 60}"#,
        Some("None"),
        Some("NoneType")
    )]
    fn an_sts_reply_needs_only_what_google_auth_reads(
        #[case] body: &str,
        #[case] token: Option<&str>,
        #[case] non_string: Option<&str>,
    ) {
        let minted = sts_reply(body, NOW).unwrap();
        assert_eq!(Some(minted.value.expose()), token);
        assert_eq!(minted.non_string, non_string);
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

        assert_eq!(account.mint().await.unwrap().value.expose(), "sa-token");
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
    #[case::missing(
        json!({"file": "/nonexistent/fz-subject.txt"}),
        PythonException::Refresh,
        "File '/nonexistent/fz-subject.txt' was not found."
    )]
    #[case::json_without_field(
        json!({"file": env!("CARGO_MANIFEST_PATH"), "format": {"type": "json", "subject_token_field_name": "t"}}),
        PythonException::Refresh,
        concat!("Unable to parse subject_token from JSON file '", env!("CARGO_MANIFEST_PATH"), "' using key 't'")
    )]
    #[tokio::test]
    async fn a_file_source_raises_what_identity_pool_raises(
        #[case] source: Value,
        #[case] exception: PythonException,
        #[case] message: &str,
    ) {
        let info = json!({"type": "external_account", "audience": AUDIENCE, "credential_source": source, "token_url": "http://127.0.0.1:9/token"});
        let loaded = identity_pool(info.as_object().unwrap()).unwrap().unwrap();
        assert_eq!(
            loaded.refresh().await.unwrap_err(),
            GoogleLoginFailure::new(exception, message)
        );
    }

    #[rstest]
    #[case::not_a_dict(json!("file"), PythonException::Malformed, "Invalid credential_source. The credential_source is not a dict.")]
    #[case::aws(json!({"environment_id": "aws1"}), PythonException::Malformed, "Invalid Identity Pool credential_source field 'environment_id'")]
    #[case::none(json!({}), PythonException::Malformed, "Missing credential_source. A 'file', 'url', or 'certificate' must be provided.")]
    #[case::two(json!({"file": "/a", "url": "http://b"}), PythonException::Malformed, "Ambiguous credential_source. 'file', 'url', and 'certificate' are mutually exclusive..")]
    #[case::bad_format(json!({"file": "/a", "format": {"type": "xml"}}), PythonException::Malformed, "Invalid credential_source format 'xml'")]
    #[case::json_without_name(json!({"file": "/a", "format": {"type": "json"}}), PythonException::Malformed, "Missing subject_token_field_name for JSON credential_source format")]
    fn an_identity_pool_source_is_checked_at_load(
        #[case] source: Value,
        #[case] exception: PythonException,
        #[case] message: &str,
    ) {
        let info =
            json!({"type": "external_account", "audience": AUDIENCE, "credential_source": source});
        assert_eq!(
            identity_pool(info.as_object().unwrap()).err(),
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
        let loaded = pluggable(info.as_object().unwrap(), Arc::new(|_| None)).unwrap();
        assert_eq!(
            loaded.refresh().await.unwrap_err(),
            GoogleLoginFailure::new(
                PythonException::Malformed,
                "Executables need to be explicitly allowed (set GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES to '1') to run."
            )
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
            Settings::from_info(info.as_object().unwrap()).unwrap(),
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

        assert_eq!(account.mint().await.unwrap().value.expose(), "exe-token");
        assert!(
            sts.requests.lock().unwrap()[0]
                .2
                .contains("&subject_token=aud-jwt&")
        );
    }

    #[rstest]
    #[case::timeout(json!({"command": "x", "timeout_millis": 1000}), "Timeout must be between 5 and 120 seconds.")]
    #[case::interactive(json!({"command": "x", "interactive_timeout_millis": 1000}), "Interactive timeout must be between 30 seconds and 30 minutes.")]
    fn executable_timeouts_are_bounded(#[case] executable: Value, #[case] message: &str) {
        let info = json!({"type": "external_account", "audience": AUDIENCE, "credential_source": {"executable": executable}});
        assert_eq!(
            pluggable(info.as_object().unwrap(), Arc::new(|_| None)).err(),
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
        assert_eq!(
            Settings::from_info(info.as_object().unwrap()).is_ok(),
            accepted
        );
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
