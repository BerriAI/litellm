//! Shared Vertex AI host-layer authentication.
//!
//! Rust counterpart of Python's `litellm/llms/vertex_ai/vertex_llm_base.py`
//! (`VertexBase`) scoped to what current routes need: resolve a credential
//! source (inline JSON, file path, `VERTEXAI_CREDENTIALS`, or ADC), build
//! Google credentials with the cloud-platform scope, cache them by content,
//! and mint OAuth access tokens. Any Vertex route (OCR today, others later)
//! should use this instead of owning Google auth.

use std::sync::{Once, OnceLock};

use google_cloud_auth::credentials::service_account::AccessSpecifier;
use google_cloud_auth::credentials::{
    external_account, impersonated, service_account, user_account, AccessTokenCredentials,
    Builder as AdcBuilder,
};
use litellm_core::cache::in_memory::InMemoryCache;
use litellm_core::error::CoreError;
use litellm_core::CoreResult;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;

use crate::config::resolve_env_reference;
use crate::constants::{
    CLOUD_PLATFORM_SCOPE, VERTEXAI_CREDENTIALS_ENV, VERTEX_CREDENTIALS_CACHE_CAPACITY,
};

type CacheKey = [u8; 32];

fn ensure_crypto_provider() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

fn adc_cache_key() -> CacheKey {
    let mut hasher = Sha256::new();
    hasher.update(b"adc");
    hasher.finalize().into()
}

fn content_cache_key(contents: &str) -> CacheKey {
    let mut hasher = Sha256::new();
    hasher.update(b"inline:");
    hasher.update(contents.trim().as_bytes());
    hasher.finalize().into()
}

pub struct VertexAiBase {
    cache: Mutex<InMemoryCache<CacheKey, AccessTokenCredentials>>,
}

impl VertexAiBase {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(InMemoryCache::new(VERTEX_CREDENTIALS_CACHE_CAPACITY)),
        }
    }

    pub fn shared() -> &'static VertexAiBase {
        static SHARED: OnceLock<VertexAiBase> = OnceLock::new();
        SHARED.get_or_init(VertexAiBase::new)
    }

    /// Mirrors Python `VertexBase.safe_get_vertex_ai_credentials`: request
    /// params (`vertex_credentials`, then `vertex_ai_credentials`) take
    /// precedence over the `VERTEXAI_CREDENTIALS` environment variable.
    pub fn resolve_credential_source(
        optional_params: &Map<String, Value>,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Option<String> {
        Self::credential_source_param(optional_params)
            .and_then(|source| resolve_env_reference(Some(&source), env_lookup))
            .or_else(|| {
                env_lookup(VERTEXAI_CREDENTIALS_ENV)
                    .map(|value| value.trim().to_string())
                    .filter(|value| !value.is_empty())
            })
    }

    fn credential_source_param(optional_params: &Map<String, Value>) -> Option<String> {
        ["vertex_credentials", "vertex_ai_credentials"]
            .iter()
            .find_map(|key| optional_params.get(*key))
            .and_then(|value| match value {
                Value::String(raw) => {
                    let trimmed = raw.trim();
                    (!trimmed.is_empty()).then(|| trimmed.to_string())
                }
                Value::Object(_) => Some(value.to_string()),
                _ => None,
            })
    }

    /// Mirrors Python `VertexBase.get_access_token`: load credentials from the
    /// given source (or ADC when absent), cache them by content, and return an
    /// OAuth access token. The Google auth crate refreshes expired tokens
    /// internally, matching Python's cached-credential refresh behavior.
    pub async fn get_access_token(&self, credentials: Option<&str>) -> CoreResult<String> {
        ensure_crypto_provider();
        let token = self
            .resolve_credentials(credentials)
            .await?
            .access_token()
            .await
            .map_err(|_| CoreError::Auth("Failed to obtain Vertex access token".to_string()))?;
        Ok(token.token)
    }

    async fn resolve_credentials(
        &self,
        credentials: Option<&str>,
    ) -> CoreResult<AccessTokenCredentials> {
        let (key, built) = match credentials {
            None => (adc_cache_key(), None),
            Some(raw) => {
                let contents = load_credentials_contents(raw).await?;
                (content_cache_key(&contents), Some(contents))
            }
        };
        if let Some(existing) = self.cache.lock().await.get(&key) {
            return Ok(existing);
        }
        let credentials = match built {
            None => build_adc_credentials()?,
            Some(contents) => build_from_json(parse_credentials_json(&contents)?)?,
        };
        Ok(self.cache.lock().await.get_or_insert(key, credentials))
    }

    #[cfg(test)]
    async fn cached_credential_count(&self) -> usize {
        self.cache.lock().await.len()
    }
}

impl Default for VertexAiBase {
    fn default() -> Self {
        Self::new()
    }
}

fn build_adc_credentials() -> CoreResult<AccessTokenCredentials> {
    AdcBuilder::default()
        .with_scopes([CLOUD_PLATFORM_SCOPE])
        .build_access_token_credentials()
        .map_err(|_| CoreError::Auth("Failed to load Vertex ADC credentials".to_string()))
}

async fn load_credentials_contents(raw: &str) -> CoreResult<String> {
    let trimmed = raw.trim();
    if trimmed.starts_with('{') {
        Ok(trimmed.to_string())
    } else {
        tokio::fs::read_to_string(trimmed)
            .await
            .map_err(|_| CoreError::Auth("Failed to read Vertex credentials file".to_string()))
    }
}

fn parse_credentials_json(contents: &str) -> CoreResult<Value> {
    serde_json::from_str(contents.trim())
        .map_err(|_| CoreError::Auth("Vertex credentials are not valid JSON".to_string()))
}

fn build_from_json(json: Value) -> CoreResult<AccessTokenCredentials> {
    let scopes = [CLOUD_PLATFORM_SCOPE];
    let credentials = match json.get("type").and_then(Value::as_str) {
        Some("service_account") => service_account::Builder::new(json)
            .with_access_specifier(AccessSpecifier::from_scopes(scopes))
            .build_access_token_credentials(),
        Some("authorized_user") => user_account::Builder::new(json)
            .with_scopes(scopes)
            .build_access_token_credentials(),
        Some("external_account") => external_account::Builder::new(json)
            .with_scopes(scopes)
            .build_access_token_credentials(),
        Some("impersonated_service_account") => impersonated::Builder::new(json)
            .with_scopes(scopes)
            .build_access_token_credentials(),
        Some(_) => {
            return Err(CoreError::Auth(
                "Unsupported Vertex credential type".to_string(),
            ))
        }
        None => {
            return Err(CoreError::Auth(
                "Vertex credentials JSON is missing the required `type` field".to_string(),
            ))
        }
    };
    credentials.map_err(|_| CoreError::Auth("Failed to load Vertex credentials".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Arc;

    fn inline_service_account_json() -> String {
        json!({
            "type": "service_account",
            "project_id": "proj-1",
            "private_key_id": "key-id",
            "private_key": "-----BEGIN PRIVATE KEY-----\nnot-a-real-key\n-----END PRIVATE KEY-----\n",
            "client_email": "sa@proj-1.iam.gserviceaccount.com"
        })
        .to_string()
    }

    #[test]
    fn content_cache_key_distinguishes_adc_from_inline_and_matches_on_repeat() {
        let adc = adc_cache_key();
        let inline = content_cache_key("{\"type\":\"service_account\"}");
        assert_ne!(adc, inline);
        assert_eq!(
            inline,
            content_cache_key("  {\"type\":\"service_account\"}  ")
        );
    }

    #[test]
    fn content_cache_key_differs_for_distinct_credential_sources() {
        let adc = adc_cache_key();
        let first = content_cache_key("{\"type\":\"service_account\",\"client_email\":\"a\"}");
        let second = content_cache_key("{\"type\":\"service_account\",\"client_email\":\"b\"}");
        assert_ne!(first, second);
        assert_ne!(adc, first);
        assert_ne!(adc, second);
    }

    #[tokio::test]
    async fn file_backed_credentials_key_tracks_content_not_path() {
        let path = std::env::temp_dir().join(format!(
            "vertex-cred-{}-{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        tokio::fs::write(
            &path,
            b"{\"type\":\"service_account\",\"client_email\":\"old\"}",
        )
        .await
        .expect("writes first credential file");
        let first = load_credentials_contents(path.to_str().expect("utf-8 path"))
            .await
            .expect("reads first credential file");

        tokio::fs::write(
            &path,
            b"{\"type\":\"service_account\",\"client_email\":\"new\"}",
        )
        .await
        .expect("rotates credential file");
        let second = load_credentials_contents(path.to_str().expect("utf-8 path"))
            .await
            .expect("reads rotated credential file");
        tokio::fs::remove_file(&path).await.ok();

        assert_ne!(
            content_cache_key(&first),
            content_cache_key(&second),
            "rotating a credential file at the same path must not reuse the old cache entry"
        );
    }

    #[tokio::test]
    async fn resolve_credentials_caches_inline_service_account_per_instance() {
        let base = VertexAiBase::new();
        let inline = inline_service_account_json();
        base.resolve_credentials(Some(&inline))
            .await
            .expect("first resolve builds credentials");
        base.resolve_credentials(Some(&inline))
            .await
            .expect("second resolve reuses cache");
        assert_eq!(base.cached_credential_count().await, 1);
    }

    #[tokio::test]
    async fn resolve_credentials_reads_service_account_from_file() {
        let path = std::env::temp_dir().join(format!(
            "vertex-file-cred-{}-{:?}.json",
            std::process::id(),
            std::thread::current().id()
        ));
        tokio::fs::write(&path, inline_service_account_json())
            .await
            .expect("writes credential file");
        let base = VertexAiBase::new();
        base.resolve_credentials(Some(path.to_str().expect("utf-8 path")))
            .await
            .expect("file-backed service account resolves");
        tokio::fs::remove_file(&path).await.ok();
        assert_eq!(base.cached_credential_count().await, 1);
    }

    #[tokio::test]
    async fn resolve_credentials_single_flight_under_concurrency() {
        let base = Arc::new(VertexAiBase::new());
        let inline = Arc::new(inline_service_account_json());
        let tasks: Vec<_> = (0..16)
            .map(|_| {
                let base = Arc::clone(&base);
                let inline = Arc::clone(&inline);
                tokio::spawn(async move {
                    base.resolve_credentials(Some(inline.as_str()))
                        .await
                        .expect("concurrent resolve succeeds");
                })
            })
            .collect();
        for task in tasks {
            task.await.expect("task completes");
        }
        assert_eq!(base.cached_credential_count().await, 1);
    }

    #[tokio::test]
    async fn build_from_json_builds_service_account_credentials() {
        build_from_json(json!({
            "type": "service_account",
            "project_id": "proj-1",
            "private_key_id": "key-id",
            "private_key": "-----BEGIN PRIVATE KEY-----\nnot-a-real-key\n-----END PRIVATE KEY-----\n",
            "client_email": "sa@proj-1.iam.gserviceaccount.com"
        }))
        .expect("service account dispatches and builds");
    }

    #[tokio::test]
    async fn build_from_json_builds_authorized_user_credentials() {
        build_from_json(json!({
            "type": "authorized_user",
            "client_id": "client-id.apps.googleusercontent.com",
            "client_secret": "client-secret",
            "refresh_token": "refresh-token"
        }))
        .expect("authorized user dispatches and builds");
    }

    #[tokio::test]
    async fn build_from_json_builds_impersonated_service_account_credentials() {
        build_from_json(json!({
            "type": "impersonated_service_account",
            "service_account_impersonation_url": "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/target@proj-1.iam.gserviceaccount.com:generateAccessToken",
            "source_credentials": {
                "type": "authorized_user",
                "client_id": "client-id.apps.googleusercontent.com",
                "client_secret": "client-secret",
                "refresh_token": "refresh-token"
            }
        }))
        .expect("impersonated service account dispatches and builds");
    }

    #[tokio::test]
    async fn build_from_json_builds_external_account_for_all_standard_source_mechanisms() {
        let base = |source: Value| {
            json!({
                "type": "external_account",
                "audience": "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/pr",
                "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
                "token_url": "https://sts.googleapis.com/v1/token",
                "credential_source": source
            })
        };

        build_from_json(base(json!({"file": "/var/run/secrets/token"})))
            .expect("file-sourced external account builds");
        build_from_json(base(json!({
            "url": "https://169.254.169.254/token",
            "headers": {"Metadata": "true"},
            "format": {"type": "json", "subject_token_field_name": "access_token"}
        })))
        .expect("url-sourced external account builds");
        build_from_json(base(json!({
            "executable": {"command": "/usr/bin/token-helper", "timeout_millis": 5000}
        })))
        .expect("executable-sourced external account builds");
        build_from_json(base(json!({
            "environment_id": "aws1",
            "region_url": "http://169.254.169.254/latest/meta-data/placement/availability-zone",
            "regional_cred_verification_url": "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15"
        })))
        .expect("aws-sourced external account builds");
    }

    #[test]
    fn build_from_json_rejects_unknown_credential_type() {
        let err =
            build_from_json(json!({"type": "totally_made_up"})).expect_err("unknown type rejected");
        assert!(matches!(err, CoreError::Auth(_)), "{err:?}");
    }

    #[test]
    fn build_from_json_requires_type_field() {
        let err = build_from_json(json!({"client_email": "x"})).expect_err("missing type rejected");
        assert!(matches!(err, CoreError::Auth(_)), "{err:?}");
    }

    #[test]
    fn build_from_json_unknown_type_error_omits_attacker_controlled_type() {
        let err = build_from_json(json!({"type": "attacker-controlled-type-string"}))
            .expect_err("unknown type rejected");
        match err {
            CoreError::Auth(message) => {
                assert!(
                    !message.contains("attacker-controlled-type-string"),
                    "{message}"
                );
            }
            other => panic!("expected auth error, got {other:?}"),
        }
    }

    #[test]
    fn parse_credentials_json_reports_invalid_json_without_echoing_contents() {
        let err = parse_credentials_json("{not-valid-json-secret-value")
            .expect_err("invalid json rejected");
        match err {
            CoreError::Auth(message) => {
                assert!(
                    !message.contains("not-valid-json-secret-value"),
                    "{message}"
                );
            }
            other => panic!("expected auth error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn load_credentials_contents_missing_file_error_omits_attacker_controlled_path() {
        let err = load_credentials_contents("/attacker/controlled/secret-credentials-path.json")
            .await
            .expect_err("missing file rejected");
        match err {
            CoreError::Auth(message) => {
                assert!(!message.contains("attacker"), "{message}");
                assert!(!message.contains("secret-credentials-path"), "{message}");
            }
            other => panic!("expected auth error, got {other:?}"),
        }
    }

    #[test]
    fn resolve_credential_source_reads_string_object_and_treats_blank_as_absent() {
        let mut inline = Map::new();
        inline.insert(
            "vertex_credentials".to_string(),
            Value::String("  /path/to/sa.json  ".into()),
        );
        assert_eq!(
            VertexAiBase::resolve_credential_source(&inline, &|_| None).as_deref(),
            Some("/path/to/sa.json")
        );

        let mut object = Map::new();
        object.insert(
            "vertex_credentials".to_string(),
            json!({"type": "service_account"}),
        );
        assert_eq!(
            VertexAiBase::resolve_credential_source(&object, &|_| None).as_deref(),
            Some("{\"type\":\"service_account\"}")
        );

        let mut blank = Map::new();
        blank.insert(
            "vertex_credentials".to_string(),
            Value::String("   ".into()),
        );
        assert_eq!(
            VertexAiBase::resolve_credential_source(&blank, &|_| None),
            None
        );

        assert_eq!(
            VertexAiBase::resolve_credential_source(&Map::new(), &|_| None),
            None
        );
    }

    #[test]
    fn resolve_credential_source_prefers_optional_param_then_env() {
        let mut params = Map::new();
        params.insert(
            "vertex_credentials".to_string(),
            Value::String("/from/param.json".into()),
        );
        let env =
            |key: &str| (key == VERTEXAI_CREDENTIALS_ENV).then(|| "/from/env.json".to_string());
        assert_eq!(
            VertexAiBase::resolve_credential_source(&params, &env).as_deref(),
            Some("/from/param.json")
        );
        assert_eq!(
            VertexAiBase::resolve_credential_source(&Map::new(), &env).as_deref(),
            Some("/from/env.json")
        );

        let blank_env = |key: &str| (key == VERTEXAI_CREDENTIALS_ENV).then(|| "   ".to_string());
        assert_eq!(
            VertexAiBase::resolve_credential_source(&Map::new(), &blank_env),
            None
        );

        assert_eq!(
            VertexAiBase::resolve_credential_source(&Map::new(), &|_| None),
            None
        );
    }

    #[test]
    fn resolve_credential_source_falls_back_through_alias_param() {
        let mut params = Map::new();
        params.insert(
            "vertex_ai_credentials".to_string(),
            Value::String("/from/alias.json".into()),
        );
        assert_eq!(
            VertexAiBase::resolve_credential_source(&params, &|_| None).as_deref(),
            Some("/from/alias.json")
        );
    }

    #[test]
    fn resolve_credential_source_resolves_exact_environment_reference() {
        let params = Map::from_iter([(
            "vertex_credentials".to_string(),
            Value::String("os.environ/CUSTOM_VERTEX_CREDENTIALS".into()),
        )]);
        let env = |key: &str| {
            (key == "CUSTOM_VERTEX_CREDENTIALS").then(|| "{\"type\":\"authorized_user\"}".into())
        };
        assert_eq!(
            VertexAiBase::resolve_credential_source(&params, &env).as_deref(),
            Some("{\"type\":\"authorized_user\"}")
        );
    }

    #[test]
    fn resolve_credential_source_treats_unresolved_reference_as_absent() {
        let params = Map::from_iter([(
            "vertex_credentials".to_string(),
            Value::String("os.environ/MISSING_VERTEX_CREDENTIALS".into()),
        )]);
        assert_eq!(
            VertexAiBase::resolve_credential_source(&params, &|_| None),
            None
        );
    }
}
