//! google-auth's `default()` for the Vertex AI Messages login: application default credentials
//! from `GOOGLE_APPLICATION_CREDENTIALS`, the gcloud credentials file or the GCE metadata
//! server, and the project they resolve.

use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use serde_json::{Map, Value};

use super::{
    authorized_user, aws_wif, builders,
    credentials::{Loaded, managed},
    environment::GoogleEnvironment,
    external_account::{self, ExternalAccountInfo, OsEnvironment},
    external_authorized_user,
    failure::{GoogleLoginFailure, PythonException},
    source::{CredentialsFile, ImpersonatedInfo, Unparsed},
};

const MISSING_CREDENTIALS: &str = "Your default credentials were not found. To set up Application Default Credentials, see https://cloud.google.com/docs/authentication/external/set-up-adc for more information.";
const VALID_TYPES: &str = "('authorized_user', 'service_account', 'external_account', 'external_account_authorized_user', 'impersonated_service_account', 'gdch_service_account')";
const CLOUD_RESOURCE_MANAGER: &str = "https://cloudresourcemanager.{universe_domain}/v1/projects/";
const AWS_SUBJECT_TOKEN_TYPE: &str = "urn:ietf:params:aws:token-type:aws4_request";
const DEFAULT_UNIVERSE_DOMAIN: &str = "googleapis.com";

/// What google-auth's `default()` found: the credentials and the project it resolved.
pub(crate) struct DefaultCredentials {
    pub(crate) source: Loaded,
    pub(crate) project: Option<String>,
}

/// Whether the credentials found are an external account, whose project google-auth looks up
/// from its audience when nothing else names one.
enum Kind {
    ExternalAccount { info: Box<ExternalAccountInfo> },
    Other,
}

/// Credentials one of `default()`'s sources found, with the project that source names.
struct Found {
    source: Loaded,
    project: Option<String>,
    kind: Kind,
}

/// A project that names something: google-auth skips an empty one.
fn named(project: &Option<String>) -> bool {
    project.as_ref().is_some_and(|project| !project.is_empty())
}

fn default_failure(message: String) -> GoogleLoginFailure {
    GoogleLoginFailure::new(PythonException::DefaultCredentials, message)
}

/// `_cloud_sdk.get_application_default_credentials_path`.
fn gcloud_credentials_path(env: &dyn GoogleEnvironment) -> Option<PathBuf> {
    let config = match env.os_env("CLOUDSDK_CONFIG") {
        Some(config) => PathBuf::from(config),
        None => PathBuf::from(env.os_env("HOME")?)
            .join(".config")
            .join("gcloud"),
    };
    Some(config.join("application_default_credentials.json"))
}

fn wrapped(
    build: Result<Loaded, GoogleLoginFailure>,
    credential: &str,
    file: &str,
) -> Result<Loaded, GoogleLoginFailure> {
    build.map_err(|cause| {
        default_failure(format!(
            "Failed to load {credential} credentials from {file}: {}",
            cause.message
        ))
    })
}

fn os_env(env: &dyn GoogleEnvironment) -> OsEnvironment {
    let variables: Vec<(String, String)> = [
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES",
    ]
    .into_iter()
    .filter_map(|name| Some((name.to_owned(), env.os_env(name)?)))
    .collect();
    Arc::new(move |name| {
        variables
            .iter()
            .find(|(known, _)| known == name)
            .map(|(_, value)| value.clone())
    })
}

/// `_default._get_external_account_credentials`: AWS and executable sources raise their own
/// failure, an identity pool's `ValueError` is named by the file it came from.
fn external_account(
    env: &dyn GoogleEnvironment,
    info: &ExternalAccountInfo,
    file: &str,
) -> Result<Loaded, GoogleLoginFailure> {
    if info.subject_token_type.as_deref() == Some(AWS_SUBJECT_TOKEN_TYPE) {
        return aws_wif::metadata_aws_credentials(info, os_env(env));
    }
    if info
        .credential_source
        .as_ref()
        .is_some_and(|source| source.executable.is_some())
    {
        return external_account::pluggable(info, os_env(env));
    }
    let unloadable = |failure: GoogleLoginFailure| match failure.exception {
        PythonException::Value
        | PythonException::Malformed
        | PythonException::InvalidValue
        | PythonException::InvalidResource => default_failure(format!(
            "Failed to load external account credentials from {file}"
        )),
        _ => failure,
    };
    match external_account::identity_pool(info).map_err(unloadable)? {
        Some(loaded) => Ok(loaded),
        None => builders::external_account(info)
            .map(managed)
            .map_err(unloadable),
    }
}

/// Impersonated service account credentials without source credentials of a supported type or
/// without an impersonation URL cannot be used.
fn unusable_impersonation(info: &ImpersonatedInfo, file: &str) -> Option<GoogleLoginFailure> {
    let supported = matches!(
        info.source_credentials
            .as_ref()
            .and_then(|source| source.kind.as_deref()),
        Some("authorized_user" | "service_account" | "external_account_authorized_user")
    );
    (!supported || info.service_account_impersonation_url.is_none()).then(|| {
        default_failure(format!(
            "Failed to load impersonated service account credentials from {file}"
        ))
    })
}

/// `_default.load_credentials_from_file`.
fn load_credentials_from_file(
    env: &dyn GoogleEnvironment,
    file: &str,
) -> Result<Found, GoogleLoginFailure> {
    if !Path::new(file).exists() {
        return Err(default_failure(format!("File {file} was not found.")));
    }
    let contents = std::fs::read_to_string(file).map_err(|error| {
        default_failure(format!("File {file} is not a valid json file. {error}"))
    })?;
    let info = match serde_json::from_str::<Value>(&contents) {
        Ok(Value::Object(info)) => info,
        Ok(_) => {
            return Err(default_failure(format!(
                "File {file} is not a valid json file."
            )));
        }
        Err(error) => {
            return Err(default_failure(format!(
                "File {file} is not a valid json file: {error}"
            )));
        }
    };
    let found = |source, project, kind| Found {
        source,
        project,
        kind,
    };
    match parse_file(info, file)? {
        CredentialsFile::AuthorizedUser(user) => Ok(found(
            wrapped(
                authorized_user::authorized_user(&user, &env.google_token_endpoint()),
                "authorized user",
                file,
            )?,
            None,
            Kind::Other,
        )),
        CredentialsFile::ServiceAccount(account) => Ok(found(
            wrapped(builders::service_account(&account), "service account", file)?,
            account.project_id,
            Kind::Other,
        )),
        CredentialsFile::ExternalAccount(account) => Ok(found(
            external_account(env, &account, file)?,
            None,
            Kind::ExternalAccount { info: account },
        )),
        CredentialsFile::ImpersonatedServiceAccount(account) => {
            if let Some(unusable) = unusable_impersonation(&account, file) {
                return Err(unusable);
            }
            Ok(found(
                wrapped(
                    builders::impersonated_service_account(&account).map(managed),
                    "impersonated service account",
                    file,
                )?,
                None,
                Kind::Other,
            ))
        }
        CredentialsFile::ExternalAccountAuthorizedUser(user) => Ok(found(
            external_authorized_user::external_account_authorized_user(&user, file)?,
            None,
            Kind::Other,
        )),
        CredentialsFile::GdchServiceAccount => Err(default_failure(format!(
            "The file {file} holds gdch_service_account credentials, which the Rust Vertex AI login does not support."
        ))),
        CredentialsFile::Unknown => Err(invalid_type(file)),
    }
}

fn invalid_type(file: &str) -> GoogleLoginFailure {
    default_failure(format!(
        "The file {file} does not have a valid type, expected one of {VALID_TYPES}."
    ))
}

/// The credentials a file holds, by the type it declares.
fn parse_file(info: Map<String, Value>, file: &str) -> Result<CredentialsFile, GoogleLoginFailure> {
    CredentialsFile::parse(info).map_err(|unparsed| match unparsed {
        Unparsed::Untyped => invalid_type(file),
        Unparsed::Field(error) => default_failure(format!(
            "File {file} holds a field of the wrong type: {error}"
        )),
    })
}

/// `_default._get_gcloud_sdk_credentials`.
async fn gcloud_sdk_credentials(
    env: &dyn GoogleEnvironment,
) -> Result<Option<Found>, GoogleLoginFailure> {
    let Some(path) = gcloud_credentials_path(env).filter(|path| path.is_file()) else {
        return Ok(None);
    };
    let found = load_credentials_from_file(env, &path.to_string_lossy())?;
    if named(&found.project) {
        return Ok(Some(found));
    }
    Ok(Some(Found {
        project: env.gcloud_project().await,
        ..found
    }))
}

/// `_default._get_explicit_environ_credentials`.
async fn explicit_environ_credentials(
    env: &dyn GoogleEnvironment,
) -> Result<Option<Found>, GoogleLoginFailure> {
    let explicit_file = env
        .os_env("GOOGLE_APPLICATION_CREDENTIALS")
        .unwrap_or_default();
    if explicit_file.is_empty() {
        return Ok(None);
    }
    if gcloud_credentials_path(env).is_some_and(|path| path == Path::new(&explicit_file)) {
        return gcloud_sdk_credentials(env).await;
    }
    load_credentials_from_file(env, &explicit_file).map(Some)
}

/// `_default._get_gce_credentials`.
async fn gce_credentials(env: &dyn GoogleEnvironment) -> Result<Option<Found>, GoogleLoginFailure> {
    if !env.on_gce().await {
        return Ok(None);
    }
    Ok(Some(Found {
        project: env.gce_project().await,
        source: managed(env.gce_credentials()?),
        kind: Kind::Other,
    }))
}

/// `external_account.Credentials.project_number`, else the workforce pool user project.
fn project_number(info: &ExternalAccountInfo) -> Option<String> {
    let audience = info.audience.as_deref().unwrap_or_default();
    let components: Vec<&str> = audience.split('/').collect();
    components
        .iter()
        .position(|component| *component == "projects")
        .and_then(|index| components.get(index + 1))
        .filter(|number| !number.is_empty())
        .map(|number| (*number).to_string())
        .or_else(|| {
            info.workforce_pool_user_project
                .clone()
                .filter(|project| !project.is_empty())
        })
}

/// `external_account.Credentials.get_project_id`.
async fn external_account_project(
    env: &dyn GoogleEnvironment,
    source: &Loaded,
    info: &ExternalAccountInfo,
) -> Result<Option<String>, GoogleLoginFailure> {
    let Some(number) = project_number(info) else {
        return Ok(None);
    };
    let token = source.refresh().await?.usable()?;
    let universe_domain = info
        .universe_domain
        .as_deref()
        .unwrap_or(DEFAULT_UNIVERSE_DOMAIN);
    let url = format!(
        "{}{number}",
        CLOUD_RESOURCE_MANAGER.replace("{universe_domain}", universe_domain)
    );
    env.resource_manager_project(url, token).await
}

/// google-auth's `default()`: `GOOGLE_APPLICATION_CREDENTIALS`, the gcloud credentials file, then
/// the GCE metadata server, with `GOOGLE_CLOUD_PROJECT` or `GCLOUD_PROJECT` naming the project
/// ahead of whatever the credentials carry.
pub(crate) async fn default_credentials(
    env: &dyn GoogleEnvironment,
) -> Result<DefaultCredentials, GoogleLoginFailure> {
    let explicit_project = env
        .os_env("GOOGLE_CLOUD_PROJECT")
        .or_else(|| env.os_env("GCLOUD_PROJECT"));
    let found = match explicit_environ_credentials(env).await? {
        Some(found) => found,
        None => match gcloud_sdk_credentials(env).await? {
            Some(found) => found,
            None => gce_credentials(env)
                .await?
                .ok_or_else(|| default_failure(MISSING_CREDENTIALS.into()))?,
        },
    };
    let project = match named(&explicit_project) {
        true => explicit_project,
        false => found.project,
    };
    let project = match (&found.kind, named(&project)) {
        (Kind::ExternalAccount { info }, false) => {
            external_account_project(env, &found.source, info).await?
        }
        _ => project,
    };
    Ok(DefaultCredentials {
        source: found.source,
        project,
    })
}
