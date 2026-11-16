//! Python's `VertexBase.load_auth` for Claude on Vertex AI's `/v1/messages`, the only route that
//! logs in through it. Vertex OCR, cache-gcs, redis-connect and model-catalog-refresh keep the
//! shared [`crate::GcpAuth`] loader.

mod adc;
mod aws_wif;
mod builders;
mod environment;
mod failure;
mod source;

use std::sync::Arc;

use litellm_auth_types::{Error, VertexParams};
use litellm_secret_value::SecretValue;
use moka::future::Cache;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

pub use environment::{EnvironmentFuture, GoogleEnvironment, SystemGoogleEnvironment};
pub use failure::{GoogleLoginFailure, PythonException};

use builders::{TokenSource, token_failure};
use failure::python_type;
use source::{CredentialsJson, FILE_FAILURE_MESSAGE, load_vertex_credentials_source};

const MAX_CACHED_LOGINS: u64 = 64;

/// Python's `a or b or ...`: the first non-empty value, else the last one, which may be empty.
pub fn python_or(values: impl IntoIterator<Item = Option<String>>) -> Option<String> {
    values.into_iter().fold(None, |chosen, value| match chosen {
        Some(chosen) if !chosen.is_empty() => Some(chosen),
        _ => value,
    })
}

/// Python's `VertexBase.safe_get_vertex_ai_credentials`, with the deployment's
/// `VERTEXAI_CREDENTIALS` read through the host's secrets as `get_secret_str` reads it.
pub fn vertex_credentials(params: &VertexParams, secrets: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    python_or(
        VertexParams::CREDENTIALS
            .wire
            .iter()
            .map(|name| params.get(name))
            .chain(VertexParams::CREDENTIALS.env.iter().map(|name| secrets(name))),
    )
}

/// Python's `VertexBase.safe_get_vertex_ai_project`. `litellm.vertex_project` reaches the params
/// as `vertex_project` when the call names no project.
pub fn vertex_project(params: &VertexParams, secrets: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    python_or(
        VertexParams::PROJECT
            .wire
            .iter()
            .map(|name| params.get(name))
            .chain(VertexParams::PROJECT.env.iter().map(|name| secrets(name))),
    )
}

/// Python's `VertexBase.safe_get_vertex_ai_location`.
pub fn vertex_location(params: &VertexParams, secrets: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    python_or(
        VertexParams::LOCATION
            .wire
            .iter()
            .map(|name| params.get(name))
            .chain(VertexParams::LOCATION.env.iter().map(|name| secrets(name))),
    )
}

/// Python's `VertexBase.get_vertex_region`: a model the model map lists `supported_regions` for
/// is sent to the first of them unless the configured location is one of them.
pub fn vertex_region(location: Option<String>, supported_regions: &[String]) -> String {
    match (supported_regions.first(), location) {
        (None, location) => location
            .filter(|location| !location.is_empty())
            .unwrap_or_else(|| "us-central1".into()),
        (Some(_), Some(location)) if supported_regions.contains(&location) => location,
        (Some(first), _) => first.clone(),
    }
}

/// The token a Vertex AI Messages call is sent with and the project its URL names.
#[derive(Clone, Debug)]
pub struct VertexLogin {
    pub token: SecretValue,
    pub project_id: String,
}

#[derive(Clone, Hash, PartialEq, Eq)]
struct LoginKey {
    credentials: Option<[u8; 32]>,
    project: Option<String>,
}

struct Login {
    source: TokenSource,
    project_id: String,
}

#[derive(Clone)]
pub struct VertexMessagesAuth {
    logins: Cache<LoginKey, Arc<Login>>,
    environment: Arc<dyn GoogleEnvironment>,
}

impl Default for VertexMessagesAuth {
    fn default() -> Self {
        Self::new(Arc::new(SystemGoogleEnvironment::default()))
    }
}

fn value_failure(message: impl Into<String>) -> GoogleLoginFailure {
    GoogleLoginFailure::new(PythonException::Value, message)
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

/// The project `load_auth` reads off credentials it built: an authorized user's quota project,
/// a service account's `project_id`, nothing for an external account.
fn credentials_project(info: &Map<String, Value>) -> Option<Value> {
    match info.get("type").and_then(Value::as_str) {
        Some("authorized_user") => info.get("quota_project_id").cloned(),
        Some("external_account") => None,
        _ => info.get("project_id").cloned(),
    }
}

impl VertexMessagesAuth {
    pub fn new(environment: Arc<dyn GoogleEnvironment>) -> Self {
        Self {
            logins: Cache::builder().max_capacity(MAX_CACHED_LOGINS).build(),
            environment,
        }
    }

    /// Python's `_ensure_access_token` for `vertex_ai`: `credentials` and `project` as
    /// `safe_get_vertex_ai_credentials` and `safe_get_vertex_ai_project` resolve them.
    pub async fn login(&self, credentials: Option<String>, project: Option<String>) -> Result<VertexLogin, Error> {
        let key = LoginKey {
            credentials: credentials.as_deref().map(|text| Sha256::digest(text).into()),
            project: project.clone(),
        };
        let login = self
            .logins
            .try_get_with(key, self.load(credentials, project))
            .await
            .map_err(|error| (*error).clone())?;
        let token = login.source.token().await.map_err(|error| Error::from(token_failure(&error)))?;
        Ok(VertexLogin {
            token,
            project_id: login.project_id.clone(),
        })
    }

    fn explicit_source(&self, credentials: &CredentialsJson) -> Result<TokenSource, GoogleLoginFailure> {
        let info = &credentials.info;
        let built = match info.get("type").and_then(Value::as_str) {
            Some("external_account") if aws_wif::uses_explicit_aws(info) => {
                let environment = self.environment.clone();
                aws_wif::explicit_aws_credentials(info, Arc::new(move |name| environment.os_env(name)))
            }
            Some("external_account") => builders::external_account(info),
            Some("authorized_user") => builders::authorized_user(info),
            _ => builders::service_account(info),
        };
        built.map_err(|failure| match credentials.from_file {
            true => value_failure(FILE_FAILURE_MESSAGE),
            false => failure,
        })
    }

    async fn load(&self, credentials: Option<String>, project: Option<String>) -> Result<Arc<Login>, Error> {
        let (source, credential_project) = match credentials {
            Some(text) => {
                let json = load_vertex_credentials_source(&text)?;
                (self.explicit_source(&json)?, credentials_project(&json.info))
            }
            None => {
                let found = adc::default_credentials(&*self.environment).await?;
                (found.source, found.project)
            }
        };
        let project = project.map(Value::String).or(credential_project);
        source.token().await.map_err(|error| Error::from(token_failure(&error)))?;
        let project_id = match project {
            project if !is_truthy(project.as_ref()) => Err(value_failure("Could not resolve project_id")),
            Some(Value::String(project)) => Ok(project),
            Some(other) => Err(GoogleLoginFailure::new(
                PythonException::Type,
                format!("Expected project_id to be a str but got {}", python_type(&other)),
            )),
            None => Err(value_failure("Could not resolve project_id")),
        }?;
        Ok(Arc::new(Login { source, project_id }))
    }
}
