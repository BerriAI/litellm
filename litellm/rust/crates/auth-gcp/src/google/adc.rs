//! google-auth's `default()`: where application default
//! credentials are found (`GOOGLE_APPLICATION_CREDENTIALS`, the gcloud credentials file or the
//! GCE metadata server), which credentials Google's Rust SDK builds from them, and the project
//! they resolve.

use std::{path::PathBuf, sync::Arc};

use serde_json::{Map, Value};

use super::{
    Login,
    builders::{self, build},
    credentials::{
        CredentialsFile, CredentialsType, ExternalAccountInfo, ImpersonatedInfo, Unparsed,
    },
    environment::GoogleEnvironment,
    failure::{GoogleLoginFailure, PythonException},
    legacy,
};
use crate::{GcpTokenSource, identity::SdkCredentials};

const MISSING_CREDENTIALS: &str = "No application default credentials were found. Set GOOGLE_APPLICATION_CREDENTIALS or run `gcloud auth application-default login`.";
const CLOUD_RESOURCE_MANAGER: &str = "https://cloudresourcemanager.{universe_domain}/v1/projects/";
const DEFAULT_UNIVERSE_DOMAIN: &str = "googleapis.com";

/// An application default credentials file, and whether it is gcloud's own, whose project
/// `gcloud config` names when the file does not.
pub(crate) struct AdcFile {
    pub(crate) path: String,
    gcloud: bool,
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

/// The file `_get_explicit_environ_credentials` or `_get_gcloud_sdk_credentials` loads, `None`
/// when `default()` would go on to the GCE metadata server.
pub(crate) fn locate(env: &dyn GoogleEnvironment) -> Option<AdcFile> {
    let gcloud = gcloud_credentials_path(env);
    match env
        .os_env("GOOGLE_APPLICATION_CREDENTIALS")
        .filter(|path| !path.is_empty())
    {
        Some(path) => Some(AdcFile {
            gcloud: gcloud.is_some_and(|gcloud| gcloud == std::path::Path::new(&path)),
            path,
        }),
        None => gcloud.filter(|path| path.is_file()).map(|path| AdcFile {
            path: path.to_string_lossy().into_owned(),
            gcloud: true,
        }),
    }
}

/// Whether the credentials found are an external account, whose project google-auth looks up
/// from its audience when nothing else names one.
enum Kind {
    ExternalAccount(Box<ExternalAccountInfo>),
    Other,
}

/// Impersonated service account credentials without source credentials of a supported type or
/// without an impersonation URL cannot be used.
fn impersonated(
    info: &ImpersonatedInfo,
    env: &Arc<dyn GoogleEnvironment>,
    file: &str,
) -> Result<Arc<dyn GcpTokenSource>, GoogleLoginFailure> {
    let unusable = || {
        default_failure(format!(
            "The impersonated service account credentials in {file} cannot be used."
        ))
    };
    let source = info.source_credentials.as_ref().ok_or_else(unusable)?;
    let url = info
        .service_account_impersonation_url
        .as_deref()
        .ok_or_else(unusable)?;
    match CredentialsType::named(source.kind.as_deref()) {
        Some(CredentialsType::AuthorizedUser | CredentialsType::ServiceAccount) => build(
            SdkCredentials::Impersonated(info.sdk.clone()),
            PythonException::DefaultCredentials,
        ),
        Some(CredentialsType::ExternalAccountAuthorizedUser) => build(
            legacy::impersonating_external_account_authorized_user(
                &source.external_authorized_user,
                url,
                env.clone(),
            )?,
            PythonException::DefaultCredentials,
        ),
        _ => Err(unusable()),
    }
}

/// `_default.load_credentials_from_file`: the credentials a file holds and the project it names.
fn load_credentials_from_file(
    env: &Arc<dyn GoogleEnvironment>,
    file: &str,
) -> Result<(Login, Kind), GoogleLoginFailure> {
    let contents = std::fs::read_to_string(file).map_err(|error| {
        default_failure(match error.kind() {
            std::io::ErrorKind::NotFound => format!("The credentials file {file} does not exist."),
            _ => format!("The credentials file {file} cannot be read: {error}"),
        })
    })?;
    let info = match serde_json::from_str::<Value>(&contents) {
        Ok(Value::Object(info)) => info,
        Ok(_) => {
            return Err(default_failure(format!(
                "The credentials file {file} is not a JSON object."
            )));
        }
        Err(error) => {
            return Err(default_failure(format!(
                "The credentials file {file} is not valid JSON: {error}"
            )));
        }
    };
    let login = |source, project| Login {
        source,
        project,
        signs: false,
    };
    let unusable = PythonException::DefaultCredentials;
    Ok(match parse_file(info, file)? {
        CredentialsFile::AuthorizedUser(user) => (
            login(
                build(
                    SdkCredentials::AuthorizedUser {
                        json: user.sdk,
                        token_uri: env.google_token_endpoint(),
                    },
                    unusable,
                )?,
                None,
            ),
            Kind::Other,
        ),
        CredentialsFile::ServiceAccount(account) => (
            Login {
                signs: true,
                ..login(
                    builders::service_account(&account, unusable)?,
                    account.project_id,
                )
            },
            Kind::Other,
        ),
        CredentialsFile::ExternalAccount(account) => (
            login(builders::external_account(&account, env, false)?, None),
            Kind::ExternalAccount(account),
        ),
        CredentialsFile::ImpersonatedServiceAccount(account) => {
            (login(impersonated(&account, env, file)?, None), Kind::Other)
        }
        CredentialsFile::ExternalAccountAuthorizedUser(user) => (
            login(
                legacy::external_account_authorized_user(&user, env.clone())?,
                None,
            ),
            Kind::Other,
        ),
        CredentialsFile::GdchServiceAccount => {
            return Err(default_failure(format!(
                "The file {file} holds gdch_service_account credentials, which the Rust Vertex AI login does not support."
            )));
        }
        CredentialsFile::Unknown => return Err(invalid_type(file)),
    })
}

fn invalid_type(file: &str) -> GoogleLoginFailure {
    default_failure(format!(
        "The file {file} does not have a valid type, expected one of {}.",
        valid_types()
    ))
}

/// The types google-auth's `default()` lists as valid, which leave out `gdch_service_account`.
fn valid_types() -> String {
    <CredentialsType as strum::VariantArray>::VARIANTS
        .iter()
        .filter(|kind| **kind != CredentialsType::GdchServiceAccount)
        .map(|kind| <&'static str>::from(*kind))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The credentials a file holds, by the type it declares.
fn parse_file(info: Map<String, Value>, file: &str) -> Result<CredentialsFile, GoogleLoginFailure> {
    CredentialsFile::parse(info).map_err(|unparsed| match unparsed {
        Unparsed::Untyped => invalid_type(file),
        Unparsed::Field(error) => default_failure(format!(
            "The credentials file {file} holds a field of the wrong type: {error}"
        )),
    })
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

/// `external_account.Credentials.get_project_id`: Cloud Resource Manager asked for the project
/// the audience numbers, with the credentials' own token.
async fn external_account_project(
    env: &dyn GoogleEnvironment,
    source: &dyn GcpTokenSource,
    info: &ExternalAccountInfo,
) -> Result<Option<String>, GoogleLoginFailure> {
    let Some(number) = project_number(info) else {
        return Ok(None);
    };
    let token = source
        .token()
        .await
        .map_err(|error| GoogleLoginFailure::token(&error, env))?;
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

/// google-auth's `default()` from `file`, or the GCE metadata server when there is none, with
/// `GOOGLE_CLOUD_PROJECT` or `GCLOUD_PROJECT` naming the project ahead of the credentials.
pub(crate) async fn default_credentials(
    env: &Arc<dyn GoogleEnvironment>,
    file: Option<AdcFile>,
) -> Result<Login, GoogleLoginFailure> {
    let explicit_project = env
        .os_env("GOOGLE_CLOUD_PROJECT")
        .or_else(|| env.os_env("GCLOUD_PROJECT"));
    let (found, kind) = match file {
        Some(AdcFile { path, gcloud }) => {
            let (found, kind) = load_credentials_from_file(env, &path)?;
            let project = match gcloud && !named(&found.project) {
                true => env.gcloud_project().await,
                false => found.project,
            };
            (Login { project, ..found }, kind)
        }
        None if env.on_gce().await => (
            Login {
                source: env.gce_credentials()?,
                project: env.gce_project().await,
                signs: false,
            },
            Kind::Other,
        ),
        None => return Err(default_failure(MISSING_CREDENTIALS.into())),
    };
    let project = match named(&explicit_project) {
        true => explicit_project,
        false => found.project,
    };
    let project = match (&kind, named(&project)) {
        (Kind::ExternalAccount(info), false) => {
            external_account_project(&**env, &*found.source, info).await?
        }
        _ => project,
    };
    Ok(Login { project, ..found })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn the_valid_types_are_listed_as_google_auth_lists_them() {
        assert_eq!(
            valid_types(),
            "authorized_user, service_account, external_account, external_account_authorized_user, impersonated_service_account"
        );
    }
}
