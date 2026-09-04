use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use google_cloud_auth::credentials::impersonated;
use google_cloud_auth::credentials::{
    AccessTokenCredentials, external_account, service_account, user_account,
};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;

use crate::Error;
use crate::auth::{
    Environment, ExpiringToken, ResolvedAuth, SecretString, TokenCache, TokenProvider,
};
use crate::constants::{
    VERTEX_AUTH_CLOUD_PLATFORM_SCOPE, VERTEX_AUTH_SDK_TOKEN_CACHE_TTL_SECS,
    VERTEX_AUTH_TOKEN_REFRESH_WINDOW_SECS,
};

static CREDENTIALS: OnceLock<Mutex<HashMap<String, AccessTokenCredentials>>> = OnceLock::new();
static TOKENS: OnceLock<TokenCache> = OnceLock::new();

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexAuth {
    pub auth: ResolvedAuth,
    pub project_id: String,
}

struct GoogleTokenProvider {
    credentials: AccessTokenCredentials,
}

#[async_trait]
impl TokenProvider for GoogleTokenProvider {
    async fn token(&self) -> Result<ExpiringToken, Error> {
        let token = self
            .credentials
            .access_token()
            .await
            .map_err(|_| Error::Auth("Vertex AI credential refresh failed".to_string()))?;
        Ok(ExpiringToken {
            token: SecretString::new(token.token),
            expires_at: Instant::now() + Duration::from_secs(VERTEX_AUTH_SDK_TOKEN_CACHE_TTL_SECS),
        })
    }
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

fn value_string<'a>(params: &'a Map<String, Value>, names: &[&str]) -> Option<&'a str> {
    names
        .iter()
        .find_map(|name| params.get(*name).and_then(Value::as_str))
        .and_then(|value| non_empty(Some(value)))
}

fn fingerprint(value: &str) -> String {
    Sha256::digest(value.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn credentials_json(value: &Value) -> Result<Option<Value>, Error> {
    match value {
        Value::Null => Ok(None),
        Value::Object(_) => Ok(Some(value.clone())),
        Value::String(raw) if raw.trim_start().starts_with('{') => serde_json::from_str(raw)
            .map(Some)
            .map_err(|_| Error::Auth("Vertex AI credentials JSON is invalid".to_string())),
        Value::String(path) => std::fs::read_to_string(path)
            .map_err(|_| Error::Auth("Vertex AI credentials file could not be read".to_string()))
            .and_then(|raw| {
                serde_json::from_str(&raw).map(Some).map_err(|_| {
                    Error::Auth("Vertex AI credentials file contains invalid JSON".to_string())
                })
            }),
        _ => Err(Error::Auth(
            "Vertex AI credentials must be a JSON object, JSON string, or file path".to_string(),
        )),
    }
}

fn build_credentials(json: Option<Value>) -> Result<AccessTokenCredentials, Error> {
    let built = match json {
        None => google_cloud_auth::credentials::Builder::default()
            .with_scopes([VERTEX_AUTH_CLOUD_PLATFORM_SCOPE])
            .build_access_token_credentials(),
        Some(value) => match value.get("type").and_then(Value::as_str) {
            Some("service_account") => service_account::Builder::new(value)
                .with_access_specifier(service_account::AccessSpecifier::from_scopes([
                    VERTEX_AUTH_CLOUD_PLATFORM_SCOPE,
                ]))
                .build_access_token_credentials(),
            Some("authorized_user") => user_account::Builder::new(value)
                .with_scopes([VERTEX_AUTH_CLOUD_PLATFORM_SCOPE])
                .build_access_token_credentials(),
            Some("external_account") => external_account::Builder::new(value)
                .with_scopes([VERTEX_AUTH_CLOUD_PLATFORM_SCOPE])
                .build_access_token_credentials(),
            Some("impersonated_service_account") => impersonated::Builder::new(value)
                .with_scopes([VERTEX_AUTH_CLOUD_PLATFORM_SCOPE])
                .build_access_token_credentials(),
            _ => {
                return Err(Error::Auth(
                    "Vertex AI credential type is unsupported".to_string(),
                ));
            }
        },
    };
    built.map_err(|_| Error::Auth("Vertex AI credentials could not be loaded".to_string()))
}

fn configured_project(
    explicit: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    non_empty(explicit).map(str::to_string).or_else(|| {
        ["VERTEXAI_PROJECT", "GOOGLE_CLOUD_PROJECT", "GCLOUD_PROJECT"]
            .iter()
            .find_map(|name| env_lookup(name))
            .filter(|value| !value.trim().is_empty())
    })
}

fn inferred_project(
    explicit: Option<&str>,
    json: Option<&Value>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    configured_project(explicit, env_lookup)
        .or_else(|| {
            json.and_then(|value| {
                value
                    .get("project_id")
                    .or_else(|| value.get("quota_project_id"))
                    .and_then(Value::as_str)
                    .and_then(|value| non_empty(Some(value)))
                    .map(str::to_string)
            })
        })
        .ok_or_else(|| {
            Error::InvalidRequest(
                "Could not resolve Vertex AI project; pass vertex_project or set VERTEXAI_PROJECT"
                    .to_string(),
            )
        })
}

pub fn resolve_vertex_project(
    params: &Map<String, Value>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, Error> {
    let explicit = value_string(params, &["vertex_project", "vertex_ai_project"]);
    if let Some(project) = configured_project(explicit, env_lookup) {
        return Ok(project);
    }
    let json = params
        .get("vertex_credentials")
        .map(credentials_json)
        .transpose()?
        .flatten();
    inferred_project(None, json.as_ref(), env_lookup)
}

async fn cached_credentials(
    key: String,
    json: Option<Value>,
) -> Result<AccessTokenCredentials, Error> {
    let credentials = CREDENTIALS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut entries = credentials.lock().await;
    if let Some(credentials) = entries.get(&key) {
        return Ok(credentials.clone());
    }
    let loaded = build_credentials(json)?;
    entries.insert(key, loaded.clone());
    Ok(loaded)
}

pub async fn resolve_vertex_auth(
    api_key: Option<&str>,
    params: &Map<String, Value>,
    environment: &dyn Environment,
) -> Result<VertexAuth, Error> {
    let explicit_project = value_string(params, &["vertex_project", "vertex_ai_project"]);
    let env_lookup = |name: &str| environment.get(name);
    let configured_project = configured_project(explicit_project, &env_lookup);
    let explicit_token = non_empty(api_key)
        .map(str::to_string)
        .or_else(|| {
            environment
                .get("VERTEX_AI_API_KEY")
                .filter(|value| !value.trim().is_empty())
        })
        .or_else(|| {
            environment
                .get("VERTEXAI_API_KEY")
                .filter(|value| !value.trim().is_empty())
        });
    let credential_value = params.get("vertex_credentials");
    let json = match (configured_project.as_ref(), explicit_token.as_ref()) {
        (Some(_), Some(_)) => None,
        _ => credential_value
            .map(credentials_json)
            .transpose()?
            .flatten(),
    };
    let project_id = configured_project
        .map(Ok)
        .unwrap_or_else(|| inferred_project(None, json.as_ref(), &env_lookup))?;
    let token = match explicit_token {
        Some(token) => SecretString::new(token),
        None => {
            let identity = credential_value
                .map(Value::to_string)
                .unwrap_or_else(|| "application-default".to_string());
            let key = fingerprint(&identity);
            let provider = GoogleTokenProvider {
                credentials: cached_credentials(key.clone(), json).await?,
            };
            TOKENS
                .get_or_init(|| {
                    TokenCache::new(Duration::from_secs(VERTEX_AUTH_TOKEN_REFRESH_WINDOW_SECS))
                })
                .token(key, &provider)
                .await?
        }
    };
    Ok(VertexAuth {
        auth: ResolvedAuth::Bearer(token),
        project_id,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn explicit_token_and_project_precede_environment_without_loading_adc() {
        let params = Map::from_iter([
            (
                "vertex_project".to_string(),
                Value::String("param-project".to_string()),
            ),
            (
                "vertex_credentials".to_string(),
                Value::String("missing-credentials.json".to_string()),
            ),
        ]);
        let env = |name: &str| match name {
            "VERTEXAI_PROJECT" => Some("env-project".to_string()),
            "VERTEX_AI_API_KEY" => Some("env-token".to_string()),
            _ => None,
        };
        let resolved = resolve_vertex_auth(Some("explicit-token"), &params, &env)
            .await
            .expect("explicit credentials resolve");
        assert_eq!(resolved.project_id, "param-project");
        assert_eq!(
            resolved.auth.credential_header(),
            (
                "Authorization".to_string(),
                "Bearer explicit-token".to_string()
            )
        );
    }

    #[test]
    fn credential_json_infers_service_account_and_authorized_user_projects() {
        let env = |_: &str| None;
        let service =
            serde_json::json!({"type": "service_account", "project_id": "service-project"});
        let user =
            serde_json::json!({"type": "authorized_user", "quota_project_id": "quota-project"});
        assert_eq!(
            inferred_project(None, Some(&service), &env).unwrap(),
            "service-project"
        );
        assert_eq!(
            inferred_project(None, Some(&user), &env).unwrap(),
            "quota-project"
        );
    }
}
