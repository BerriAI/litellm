use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use super::{
    builders::{self, TokenSource, token_failure},
    environment::GoogleEnvironment,
    failure::{GoogleLoginFailure, PythonException, python_str},
};

const MISSING_CREDENTIALS: &str = "Your default credentials were not found. To set up Application Default Credentials, see https://cloud.google.com/docs/authentication/external/set-up-adc for more information.";
const VALID_TYPES: &str = "('authorized_user', 'service_account', 'external_account', 'external_account_authorized_user', 'impersonated_service_account', 'gdch_service_account')";
const CLOUD_RESOURCE_MANAGER: &str = "https://cloudresourcemanager.{universe_domain}/v1/projects/";
const AWS_SUBJECT_TOKEN_TYPE: &str = "urn:ietf:params:aws:token-type:aws4_request";
const DEFAULT_UNIVERSE_DOMAIN: &str = "googleapis.com";

/// What google-auth's `default()` found: the credentials and the project it resolved, which is
/// the JSON value a credentials file holds and may not be a string.
pub(crate) struct DefaultCredentials {
    pub(crate) source: TokenSource,
    pub(crate) project: Option<Value>,
}

enum Kind {
    ExternalAccount { info: Map<String, Value> },
    Other,
}

struct Found {
    source: TokenSource,
    project: Option<Value>,
    kind: Kind,
}

fn default_failure(message: String) -> GoogleLoginFailure {
    GoogleLoginFailure::new(PythonException::DefaultCredentials, message)
}

fn is_truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::String(text)) => !text.is_empty(),
        Some(Value::Bool(flag)) => *flag,
        Some(Value::Number(number)) => number.as_f64().is_some_and(|number| number != 0.0),
        Some(Value::Array(items)) => !items.is_empty(),
        Some(Value::Object(fields)) => !fields.is_empty(),
    }
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
    build: Result<TokenSource, GoogleLoginFailure>,
    credential: &str,
    file: &str,
) -> Result<TokenSource, GoogleLoginFailure> {
    build.map_err(|cause| {
        GoogleLoginFailure::caused_by(
            PythonException::DefaultCredentials,
            &format!("Failed to load {credential} credentials from {file}"),
            &cause,
        )
    })
}

/// `_default._get_external_account_credentials`: AWS and executable sources raise their own
/// failure, an identity pool's is named by the file it came from.
fn external_account(
    info: &Map<String, Value>,
    file: &str,
) -> Result<TokenSource, GoogleLoginFailure> {
    let raises_its_own = info.get("subject_token_type").and_then(Value::as_str)
        == Some(AWS_SUBJECT_TOKEN_TYPE)
        || info
            .get("credential_source")
            .and_then(|source| source.get("executable"))
            .is_some_and(|executable| !executable.is_null());
    builders::external_account(info).map_err(|failure| match raises_its_own {
        true => failure,
        false => default_failure(format!(
            "Failed to load external account credentials from {file}"
        )),
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
                "('File {file} is not a valid json file.', JSONDecodeError('{error}'))"
            )));
        }
    };
    let found = |source, project, kind| Found {
        source,
        project,
        kind,
    };
    match info.get("type").and_then(Value::as_str) {
        Some("authorized_user") => Ok(found(
            wrapped(
                builders::authorized_user(&info, &env.google_token_endpoint()),
                "authorized user",
                file,
            )?,
            None,
            Kind::Other,
        )),
        Some("service_account") => Ok(found(
            wrapped(builders::service_account(&info), "service account", file)?,
            info.get("project_id").cloned(),
            Kind::Other,
        )),
        Some("external_account") => Ok(found(
            external_account(&info, file)?,
            None,
            Kind::ExternalAccount { info },
        )),
        Some("impersonated_service_account") => Ok(found(
            wrapped(
                builders::impersonated_service_account(&info),
                "impersonated service account",
                file,
            )?,
            None,
            Kind::Other,
        )),
        Some(unsupported @ ("external_account_authorized_user" | "gdch_service_account")) => {
            Err(default_failure(format!(
                "The file {file} holds {unsupported} credentials, which the Rust Vertex AI login does not support."
            )))
        }
        _ => Err(default_failure(format!(
            "The file {file} does not have a valid type. Type is {}, expected one of {VALID_TYPES}.",
            python_str(info.get("type"))
        ))),
    }
}

/// `_default._get_gcloud_sdk_credentials`.
async fn gcloud_sdk_credentials(
    env: &dyn GoogleEnvironment,
) -> Result<Option<Found>, GoogleLoginFailure> {
    let Some(path) = gcloud_credentials_path(env).filter(|path| path.is_file()) else {
        return Ok(None);
    };
    let found = load_credentials_from_file(env, &path.to_string_lossy())?;
    if is_truthy(found.project.as_ref()) {
        return Ok(Some(found));
    }
    Ok(Some(Found {
        project: env.gcloud_project().await.map(Value::String),
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
        project: env.gce_project().await.map(Value::String),
        source: env.gce_credentials()?,
        kind: Kind::Other,
    }))
}

/// `external_account.Credentials.project_number`, else the workforce pool user project.
fn project_number(info: &Map<String, Value>) -> Option<String> {
    let audience = info
        .get("audience")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let components: Vec<&str> = audience.split('/').collect();
    components
        .iter()
        .position(|component| *component == "projects")
        .and_then(|index| components.get(index + 1))
        .filter(|number| !number.is_empty())
        .map(|number| (*number).to_string())
        .or_else(|| {
            info.get("workforce_pool_user_project")
                .and_then(Value::as_str)
                .filter(|project| !project.is_empty())
                .map(str::to_string)
        })
}

/// `external_account.Credentials.get_project_id`.
async fn external_account_project(
    env: &dyn GoogleEnvironment,
    source: &TokenSource,
    info: &Map<String, Value>,
) -> Result<Option<Value>, GoogleLoginFailure> {
    let Some(number) = project_number(info) else {
        return Ok(None);
    };
    let token = source
        .token()
        .await
        .map_err(|error| token_failure(&error))?;
    let universe_domain = info
        .get("universe_domain")
        .and_then(Value::as_str)
        .unwrap_or(DEFAULT_UNIVERSE_DOMAIN);
    let url = format!(
        "{}{number}",
        CLOUD_RESOURCE_MANAGER.replace("{universe_domain}", universe_domain)
    );
    Ok(env
        .resource_manager_project(url, token)
        .await?
        .map(Value::String))
}

/// google-auth's `default()`: `GOOGLE_APPLICATION_CREDENTIALS`, the gcloud credentials file, then
/// the GCE metadata server, with `GOOGLE_CLOUD_PROJECT` or `GCLOUD_PROJECT` naming the project
/// ahead of whatever the credentials carry.
pub(crate) async fn default_credentials(
    env: &dyn GoogleEnvironment,
) -> Result<DefaultCredentials, GoogleLoginFailure> {
    let explicit_project = env
        .os_env("GOOGLE_CLOUD_PROJECT")
        .or_else(|| env.os_env("GCLOUD_PROJECT"))
        .map(Value::String);
    let found = match explicit_environ_credentials(env).await? {
        Some(found) => found,
        None => match gcloud_sdk_credentials(env).await? {
            Some(found) => found,
            None => gce_credentials(env)
                .await?
                .ok_or_else(|| default_failure(MISSING_CREDENTIALS.into()))?,
        },
    };
    let project = match is_truthy(explicit_project.as_ref()) {
        true => explicit_project,
        false => found.project,
    };
    let project = match (&found.kind, is_truthy(project.as_ref())) {
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
