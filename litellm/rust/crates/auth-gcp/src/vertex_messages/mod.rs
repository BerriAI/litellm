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
pub fn vertex_credentials(
    params: &VertexParams,
    secrets: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    python_or(
        VertexParams::CREDENTIALS
            .wire
            .iter()
            .map(|name| params.get(name))
            .chain(
                VertexParams::CREDENTIALS
                    .env
                    .iter()
                    .map(|name| secrets(name)),
            ),
    )
}

/// Python's `VertexBase.safe_get_vertex_ai_project`. `litellm.vertex_project` reaches the params
/// as `vertex_project` when the call names no project.
pub fn vertex_project(
    params: &VertexParams,
    secrets: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    python_or(
        VertexParams::PROJECT
            .wire
            .iter()
            .map(|name| params.get(name))
            .chain(VertexParams::PROJECT.env.iter().map(|name| secrets(name))),
    )
}

/// Python's `VertexBase.safe_get_vertex_ai_location`.
pub fn vertex_location(
    params: &VertexParams,
    secrets: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
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
    pub async fn login(
        &self,
        credentials: Option<String>,
        project: Option<String>,
    ) -> Result<VertexLogin, Error> {
        let key = LoginKey {
            credentials: credentials
                .as_deref()
                .map(|text| Sha256::digest(text).into()),
            project: project.clone(),
        };
        let login = self
            .logins
            .try_get_with(key, self.load(credentials, project))
            .await
            .map_err(|error| (*error).clone())?;
        let token = login
            .source
            .token()
            .await
            .map_err(|error| Error::from(token_failure(&error)))?;
        Ok(VertexLogin {
            token,
            project_id: login.project_id.clone(),
        })
    }

    fn explicit_source(
        &self,
        credentials: &CredentialsJson,
    ) -> Result<TokenSource, GoogleLoginFailure> {
        let info = &credentials.info;
        let built = match info.get("type").and_then(Value::as_str) {
            Some("external_account") if aws_wif::uses_explicit_aws(info) => {
                let environment = self.environment.clone();
                aws_wif::explicit_aws_credentials(
                    info,
                    Arc::new(move |name| environment.os_env(name)),
                )
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

    async fn load(
        &self,
        credentials: Option<String>,
        project: Option<String>,
    ) -> Result<Arc<Login>, Error> {
        let (source, credential_project) = match credentials {
            Some(text) => {
                let json = load_vertex_credentials_source(&text)?;
                (
                    self.explicit_source(&json)?,
                    credentials_project(&json.info),
                )
            }
            None => {
                let found = adc::default_credentials(&*self.environment).await?;
                (found.source, found.project)
            }
        };
        let project = project.map(Value::String).or(credential_project);
        source
            .token()
            .await
            .map_err(|error| Error::from(token_failure(&error)))?;
        let project_id = match project {
            project if !is_truthy(project.as_ref()) => {
                Err(value_failure("Could not resolve project_id"))
            }
            Some(Value::String(project)) => Ok(project),
            Some(other) => Err(GoogleLoginFailure::new(
                PythonException::Type,
                format!(
                    "Expected project_id to be a str but got {}",
                    python_type(&other)
                ),
            )),
            None => Err(value_failure("Could not resolve project_id")),
        }?;
        Ok(Arc::new(Login { source, project_id }))
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use axum::{Json, Router, routing::post};
    use rstest::{fixture, rstest};
    use serde_json::json;

    use super::*;
    use crate::{
        CredentialSource, GcpAuth, VertexAuth, VertexConfig,
        identity::{GcpAuthFuture, GcpTokenSource},
    };

    struct FixedToken;

    impl GcpTokenSource for FixedToken {
        fn project_id(&self) -> GcpAuthFuture<'_, String> {
            Box::pin(async { unreachable!("the login never asks a token source for its project") })
        }

        fn token(&self) -> GcpAuthFuture<'_, SecretValue> {
            Box::pin(async { Ok(SecretValue::new("metadata-token")) })
        }
    }

    #[derive(Default)]
    struct FakeEnvironment {
        env: BTreeMap<&'static str, String>,
        gcloud_project: Option<&'static str>,
        on_gce: bool,
        gce_project: Option<&'static str>,
        resource_manager_project: Option<&'static str>,
        resource_manager_urls: std::sync::Mutex<Vec<String>>,
    }

    impl GoogleEnvironment for FakeEnvironment {
        fn os_env(&self, name: &str) -> Option<String> {
            self.env.get(name).cloned()
        }

        fn gcloud_project(&self) -> EnvironmentFuture<'_, Option<String>> {
            Box::pin(async { self.gcloud_project.map(str::to_string) })
        }

        fn on_gce(&self) -> EnvironmentFuture<'_, bool> {
            Box::pin(async { self.on_gce })
        }

        fn gce_project(&self) -> EnvironmentFuture<'_, Option<String>> {
            Box::pin(async { self.gce_project.map(str::to_string) })
        }

        fn gce_credentials(&self) -> Result<TokenSource, GoogleLoginFailure> {
            Ok(Arc::new(FixedToken))
        }

        fn resource_manager_project(
            &self,
            url: String,
            _token: SecretValue,
        ) -> EnvironmentFuture<'_, Result<Option<String>, GoogleLoginFailure>> {
            self.resource_manager_urls.lock().unwrap().push(url);
            Box::pin(async { Ok(self.resource_manager_project.map(str::to_string)) })
        }
    }

    struct TokenServer {
        url: String,
        requests: Arc<AtomicUsize>,
        server: tokio::task::JoinHandle<()>,
    }

    impl Drop for TokenServer {
        fn drop(&mut self) {
            self.server.abort();
        }
    }

    #[fixture]
    async fn token_server() -> TokenServer {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let requests = Arc::new(AtomicUsize::new(0));
        let recorded = requests.clone();
        let app = Router::new().route(
            "/token",
            post(move || {
                let recorded = recorded.clone();
                async move {
                    recorded.fetch_add(1, Ordering::SeqCst);
                    Json(json!({
                        "access_token": "user-token", "expires_in": 3600, "token_type": "Bearer",
                        "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
                    }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        TokenServer {
            url: format!("http://{address}/token"),
            requests,
            server,
        }
    }

    fn authorized_user(server: &TokenServer, extra: Value) -> Value {
        let base = json!({
            "type": "authorized_user", "client_id": "client", "client_secret": "secret",
            "refresh_token": "refresh", "token_uri": server.url,
        });
        Value::Object(
            base.as_object()
                .unwrap()
                .clone()
                .into_iter()
                .chain(extra.as_object().unwrap().clone())
                .collect(),
        )
    }

    fn auth(environment: FakeEnvironment) -> VertexMessagesAuth {
        VertexMessagesAuth::new(Arc::new(environment))
    }

    fn failure(error: Error) -> GoogleLoginFailure {
        litellm_error::find::<GoogleLoginFailure>(&error)
            .cloned()
            .unwrap_or_else(|| panic!("not a login failure: {error:?}"))
    }

    fn key_file(contents: &Value) -> tempfile::NamedTempFile {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), contents.to_string()).unwrap();
        file
    }

    #[rstest]
    #[case::configured_project_wins(json!({"quota_project_id": "quota"}), Some("configured"), "configured")]
    #[case::authorized_user_quota_project(json!({"quota_project_id": "quota"}), None, "quota")]
    #[tokio::test]
    async fn explicit_credentials_name_the_project_as_load_auth_does(
        #[future] token_server: TokenServer,
        #[case] extra: Value,
        #[case] configured: Option<&str>,
        #[case] expected: &str,
    ) {
        let server = token_server.await;
        let login = auth(FakeEnvironment::default())
            .login(
                Some(authorized_user(&server, extra).to_string()),
                configured.map(str::to_string),
            )
            .await
            .unwrap();
        assert_eq!(login.project_id, expected);
        assert_eq!(login.token.expose(), "user-token");
    }

    #[rstest]
    #[case::no_quota_project(json!({}), PythonException::Value, "Could not resolve project_id")]
    #[case::null_quota_project(json!({"quota_project_id": null}), PythonException::Value, "Could not resolve project_id")]
    #[case::numeric_quota_project(
        json!({"quota_project_id": 7}),
        PythonException::Type,
        "Expected project_id to be a str but got <class 'int'>"
    )]
    #[tokio::test]
    async fn an_unresolvable_project_fails_after_the_token_is_minted(
        #[future] token_server: TokenServer,
        #[case] extra: Value,
        #[case] exception: PythonException,
        #[case] message: &str,
    ) {
        let server = token_server.await;
        let error = auth(FakeEnvironment::default())
            .login(Some(authorized_user(&server, extra).to_string()), None)
            .await
            .unwrap_err();
        assert_eq!(failure(error), GoogleLoginFailure::new(exception, message));
        assert_eq!(server.requests.load(Ordering::SeqCst), 1);
    }

    #[rstest]
    #[tokio::test]
    async fn an_empty_configured_project_is_not_replaced_by_the_credentials_project(
        #[future] token_server: TokenServer,
    ) {
        let server = token_server.await;
        let error = auth(FakeEnvironment::default())
            .login(
                Some(authorized_user(&server, json!({"quota_project_id": "quota"})).to_string()),
                Some(String::new()),
            )
            .await
            .unwrap_err();
        assert_eq!(failure(error).message, "Could not resolve project_id");
    }

    #[rstest]
    #[case::impersonated(json!({
        "type": "impersonated_service_account", "client_email": "sa@p.iam.gserviceaccount.com",
        "source_credentials": {"type": "authorized_user"},
        "service_account_impersonation_url": "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/sa:generateAccessToken"
    }))]
    #[case::unknown_type(json!({"type": "gdch_service_account", "client_email": "sa@p.iam.gserviceaccount.com"}))]
    #[tokio::test]
    async fn explicit_credentials_of_another_type_are_refused_as_a_service_account(
        #[case] configured: Value,
    ) {
        let error = auth(FakeEnvironment::default())
            .login(Some(configured.to_string()), Some("project".into()))
            .await
            .unwrap_err();
        assert_eq!(
            failure(error),
            GoogleLoginFailure::new(
                PythonException::Malformed,
                "Service account info was not in the expected format, missing fields token_uri."
            )
        );
    }

    #[rstest]
    #[tokio::test]
    async fn credentials_that_do_not_build_from_a_file_name_the_file_failure() {
        let file = key_file(&json!({"type": "impersonated_service_account"}));
        let error = auth(FakeEnvironment::default())
            .login(
                Some(file.path().to_str().unwrap().into()),
                Some("project".into()),
            )
            .await
            .unwrap_err();
        assert_eq!(
            failure(error),
            GoogleLoginFailure::new(PythonException::Value, FILE_FAILURE_MESSAGE)
        );
    }

    fn environment(env: &[(&'static str, &str)]) -> FakeEnvironment {
        FakeEnvironment {
            env: env
                .iter()
                .map(|(name, value)| (*name, value.to_string()))
                .collect(),
            ..FakeEnvironment::default()
        }
    }

    #[rstest]
    #[case::credentials_file_project(&[], "file-project")]
    #[case::google_cloud_project_first(&[("GOOGLE_CLOUD_PROJECT", "env-project")], "env-project")]
    #[case::legacy_project_name(&[("GCLOUD_PROJECT", "legacy-project")], "legacy-project")]
    #[case::empty_project_variable_is_skipped(&[("GOOGLE_CLOUD_PROJECT", "")], "file-project")]
    #[tokio::test]
    async fn application_default_credentials_name_the_project_as_google_auth_does(
        #[future] token_server: TokenServer,
        #[case] env: &[(&'static str, &str)],
        #[case] expected: &str,
    ) {
        let server = token_server.await;
        let file = key_file(&authorized_user(
            &server,
            json!({"quota_project_id": "quota"}),
        ));
        let gcloud = tempfile::tempdir().unwrap();
        let path = gcloud.path().join("application_default_credentials.json");
        std::fs::copy(file.path(), &path).unwrap();
        let fake = FakeEnvironment {
            gcloud_project: Some("file-project"),
            ..environment(
                &env.iter()
                    .copied()
                    .chain([("CLOUDSDK_CONFIG", gcloud.path().to_str().unwrap())])
                    .collect::<Vec<_>>(),
            )
        };
        let login = auth(fake).login(None, None).await.unwrap();
        assert_eq!(login.project_id, expected);
    }

    #[rstest]
    #[tokio::test]
    async fn an_authorized_user_named_by_google_application_credentials_carries_no_project(
        #[future] token_server: TokenServer,
    ) {
        let server = token_server.await;
        let file = key_file(&authorized_user(
            &server,
            json!({"quota_project_id": "quota"}),
        ));
        let fake = FakeEnvironment {
            gcloud_project: Some("gcloud-project"),
            ..environment(&[(
                "GOOGLE_APPLICATION_CREDENTIALS",
                file.path().to_str().unwrap(),
            )])
        };
        let error = auth(fake).login(None, None).await.unwrap_err();
        assert_eq!(failure(error).message, "Could not resolve project_id");
    }

    #[rstest]
    #[tokio::test]
    async fn the_metadata_server_names_the_project_on_gce() {
        let fake = FakeEnvironment {
            on_gce: true,
            gce_project: Some("gce-project"),
            ..FakeEnvironment::default()
        };
        let login = auth(fake).login(None, None).await.unwrap();
        assert_eq!(login.project_id, "gce-project");
        assert_eq!(login.token.expose(), "metadata-token");
    }

    #[rstest]
    #[case::project_found(Some("crm-project"), Ok("crm-project"))]
    #[case::project_not_found(None, Err("Could not resolve project_id"))]
    #[tokio::test]
    async fn an_external_account_default_asks_resource_manager_for_its_project(
        #[future] token_server: TokenServer,
        #[case] answered: Option<&'static str>,
        #[case] expected: Result<&str, &str>,
    ) {
        let server = token_server.await;
        let subject = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(subject.path(), "subject-token").unwrap();
        let file = key_file(&json!({
            "type": "external_account",
            "audience": "//iam.googleapis.com/projects/123456/locations/global/workloadIdentityPools/p/providers/q",
            "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
            "token_url": server.url,
            "credential_source": {"file": subject.path().to_str().unwrap()},
        }));
        let fake = Arc::new(FakeEnvironment {
            resource_manager_project: answered,
            ..environment(&[(
                "GOOGLE_APPLICATION_CREDENTIALS",
                file.path().to_str().unwrap(),
            )])
        });
        let login = VertexMessagesAuth::new(fake.clone())
            .login(None, None)
            .await;
        assert_eq!(
            login
                .map(|login| login.project_id)
                .map_err(|error| failure(error).message),
            expected.map(str::to_string).map_err(str::to_string)
        );
        assert_eq!(
            *fake.resource_manager_urls.lock().unwrap(),
            ["https://cloudresourcemanager.googleapis.com/v1/projects/123456"]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_configured_project_skips_resource_manager(#[future] token_server: TokenServer) {
        let server = token_server.await;
        let subject = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(subject.path(), "subject-token").unwrap();
        let file = key_file(&json!({
            "type": "external_account",
            "audience": "//iam.googleapis.com/projects/123456/locations/global/workloadIdentityPools/p/providers/q",
            "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
            "token_url": server.url,
            "credential_source": {"file": subject.path().to_str().unwrap()},
        }));
        let fake = Arc::new(environment(&[
            (
                "GOOGLE_APPLICATION_CREDENTIALS",
                file.path().to_str().unwrap(),
            ),
            ("GOOGLE_CLOUD_PROJECT", "env-project"),
        ]));
        let login = VertexMessagesAuth::new(fake.clone())
            .login(None, None)
            .await
            .unwrap();
        assert_eq!(login.project_id, "env-project");
        assert!(fake.resource_manager_urls.lock().unwrap().is_empty());
    }

    #[rstest]
    #[case::missing_file(
        &[("GOOGLE_APPLICATION_CREDENTIALS", "/nonexistent/adc.json")],
        "File /nonexistent/adc.json was not found."
    )]
    #[case::nothing_found(
        &[],
        "Your default credentials were not found. To set up Application Default Credentials, see https://cloud.google.com/docs/authentication/external/set-up-adc for more information."
    )]
    #[tokio::test]
    async fn application_default_failures_are_default_credentials_errors(
        #[case] env: &[(&'static str, &str)],
        #[case] message: &str,
    ) {
        let error = auth(environment(env))
            .login(None, Some("p".into()))
            .await
            .unwrap_err();
        assert_eq!(
            failure(error),
            GoogleLoginFailure::new(PythonException::DefaultCredentials, message)
        );
    }

    #[rstest]
    #[tokio::test]
    async fn an_application_default_file_of_an_unknown_type_names_the_valid_types() {
        let file = key_file(&json!({"type": "weird"}));
        let path = file.path().to_str().unwrap().to_string();
        let error = auth(environment(&[("GOOGLE_APPLICATION_CREDENTIALS", &path)]))
            .login(None, Some("p".into()))
            .await
            .unwrap_err();
        assert_eq!(
            failure(error).message,
            format!(
                "The file {path} does not have a valid type. Type is weird, expected one of ('authorized_user', 'service_account', 'external_account', 'external_account_authorized_user', 'impersonated_service_account', 'gdch_service_account')."
            )
        );
    }

    #[rstest]
    #[tokio::test]
    async fn an_application_default_impersonated_service_account_is_loaded() {
        let file = key_file(&json!({
            "type": "impersonated_service_account",
            "service_account_impersonation_url": "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/sa:generateAccessToken",
            "source_credentials": {"type": "authorized_user", "client_id": "c", "client_secret": "s", "refresh_token": "r", "token_uri": "http://127.0.0.1:9/token"}
        }));
        let error = auth(environment(&[(
            "GOOGLE_APPLICATION_CREDENTIALS",
            file.path().to_str().unwrap(),
        )]))
        .login(None, Some("p".into()))
        .await
        .unwrap_err();
        assert_ne!(
            failure(error).exception,
            PythonException::DefaultCredentials
        );
    }

    #[rstest]
    #[case::param_first(Some("param"), Some("alias"), Some("secret"), Some("param"))]
    #[case::alias_next(None, Some("alias"), Some("secret"), Some("alias"))]
    #[case::secret_last(Some(""), None, Some("secret"), Some("secret"))]
    #[case::empty_secret_is_kept(None, None, Some(""), Some(""))]
    #[case::nothing(None, None, None, None)]
    fn credentials_resolve_as_python_s_or_chain(
        #[case] primary: Option<&str>,
        #[case] alias: Option<&str>,
        #[case] secret: Option<&str>,
        #[case] expected: Option<&str>,
    ) {
        let params = VertexParams {
            vertex_credentials: primary.map(Into::into),
            vertex_ai_credentials: alias.map(Into::into),
            ..VertexParams::default()
        };
        let secrets = |name: &str| {
            (name == "VERTEXAI_CREDENTIALS")
                .then(|| secret.map(str::to_string))
                .flatten()
        };
        assert_eq!(vertex_credentials(&params, &secrets).as_deref(), expected);
    }

    #[rstest]
    #[case::no_regions(Some("europe-west4"), &[], "europe-west4")]
    #[case::default_location(None, &[], "us-central1")]
    #[case::empty_location(Some(""), &[], "us-central1")]
    #[case::first_region_without_location(None, &["us-east5", "europe-west1"], "us-east5")]
    #[case::supported_location_kept(Some("europe-west1"), &["us-east5", "europe-west1"], "europe-west1")]
    #[case::unsupported_location_replaced(Some("asia-east1"), &["us-east5"], "us-east5")]
    fn the_region_follows_the_model_s_supported_regions(
        #[case] location: Option<&str>,
        #[case] supported: &[&str],
        #[case] expected: &str,
    ) {
        let supported: Vec<String> = supported.iter().map(|region| region.to_string()).collect();
        assert_eq!(
            vertex_region(location.map(str::to_string), &supported),
            expected
        );
    }

    #[rstest]
    #[tokio::test]
    async fn the_shared_loader_keeps_main_s_login_for_other_consumers(
        #[future] token_server: TokenServer,
    ) {
        let server = token_server.await;
        let credentials =
            authorized_user(&server, json!({"quota_project_id": "quota"})).to_string();
        let shared = VertexAuth::default()
            .validate_environment(
                Vec::new(),
                None,
                &VertexConfig::new(
                    Some(litellm_auth_types::Sourced::new(
                        SecretValue::new(credentials.clone()),
                        litellm_auth_types::InputSource::Deployment,
                    )),
                    None,
                    None,
                ),
                &|_| None,
            )
            .await;
        assert!(matches!(
            shared,
            Err(Error::CredentialAcquisition(
                litellm_auth_types::ErrorDetail::Unresolved {
                    subject: "GCP resource project"
                }
            ))
        ));
        let messages = auth(FakeEnvironment::default())
            .login(Some(credentials), None)
            .await
            .unwrap();
        assert_eq!(messages.project_id, "quota");
    }

    #[rstest]
    #[tokio::test]
    async fn the_shared_loader_ignores_explicit_aws_keys() {
        let explicit_aws = json!({
            "type": "external_account",
            "audience": "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/aws",
            "subject_token_type": "urn:ietf:params:aws:token-type:aws4_request",
            "token_url": "https://sts.googleapis.com/v1/token",
            "credential_source": {
                "environment_id": "aws1",
                "regional_cred_verification_url": "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15"
            },
            "aws_access_key_id": "key"
        })
        .to_string();
        let shared = GcpAuth::default()
            .load_provider(CredentialSource::Trusted(SecretValue::new(
                explicit_aws.clone(),
            )))
            .await;
        assert!(shared.is_ok(), "{:?}", shared.err());
        let error = auth(FakeEnvironment::default())
            .login(Some(explicit_aws), Some("p".into()))
            .await
            .unwrap_err();
        assert_eq!(
            failure(error),
            GoogleLoginFailure::new(
                PythonException::Value,
                "aws_region_name is required in the WIF credential JSON when using explicit AWS authentication. Add \"aws_region_name\": \"<your-region>\" to your credential file."
            )
        );
    }
}
