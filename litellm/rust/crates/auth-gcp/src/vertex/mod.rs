//! Python's `VertexBase.load_auth`, shared by every Vertex AI route: which credentials
//! `vertex_credentials`, `vertex_project` and `vertex_location` name, and the login itself, built
//! on [`crate::google`] and the shared [`crate::GcpAuth`] loader.

mod credentials_source;
mod explicit_aws;

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use litellm_auth_aws::AwsAuthService;
use litellm_auth_types::{Error, VertexParams};
use litellm_python_compat::truthy::first_truthy;
use litellm_secret_value::SecretValue;

use crate::{
    GcpAuth, GcpTokenSource,
    google::{
        GoogleEnvironment, GoogleLoginFailure, Login, PythonException, SystemGoogleEnvironment,
        UnusableFile, adc, builders,
    },
    identity::{CredentialCacheKey, SdkCredentials},
};
use credentials_source::{CredentialsJson, ExplicitCredentials, load_vertex_credentials_source};
use explicit_aws::{explicit_aws_login, uses_explicit_aws};

/// Python's `VertexBase.safe_get_vertex_ai_credentials`, with the deployment's
/// `VERTEXAI_CREDENTIALS` read through the host's secrets as `get_secret_str` reads it.
pub fn vertex_credentials(
    params: &VertexParams,
    secrets: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    first_truthy(
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
    first_truthy(
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
    first_truthy(
        VertexParams::LOCATION
            .wire
            .iter()
            .map(|name| params.get(name))
            .chain(VertexParams::LOCATION.env.iter().map(|name| secrets(name))),
    )
}

/// The token a Vertex AI call is sent with and the project its URL names.
#[derive(Clone, Debug)]
pub struct VertexLogin {
    pub token: SecretValue,
    pub project_id: String,
}

/// How many key file paths keep the credentials they loaded last.
const EARLIER_KEY_FILES: u64 = 64;

/// How long a token request may take, as `requests` waits for google-auth's token requests.
const TOKEN_TIMEOUT: Duration = Duration::from_secs(120);

/// The Vertex AI login: credentials loaded once per credentials text or key file
/// version through the shared loader, and tokens served by Google's Rust SDK.
#[derive(Clone)]
pub struct VertexLoginService {
    gcp: GcpAuth,
    environment: Arc<dyn GoogleEnvironment>,
    /// The AWS login behind an external account's explicit `aws_*` keys, which resolves an
    /// `aws_web_identity_token` reference such as `oidc/...` when it was given a resolver.
    aws: AwsAuthService,
    /// The credentials each key file path loaded last, served while a rewrite of the file does
    /// not load, as Python keeps its earlier credentials.
    earlier_key_files: moka::future::Cache<String, Arc<dyn GcpTokenSource>>,
}

impl Default for VertexLoginService {
    fn default() -> Self {
        Self::new(Arc::new(SystemGoogleEnvironment))
    }
}

fn value_failure(message: impl Into<String>) -> GoogleLoginFailure {
    GoogleLoginFailure::new(PythonException::Value, message)
}

/// The text after "Error:" when a failure carries no login failure's message. The SDK's own error
/// text can quote the credentials, so it is never logged.
const UNDESCRIBED_FAILURE: &str = "Vertex: the credentials could not be loaded.";

/// Python's `_load_auth_or_earlier_key_file`, which logs a `load_auth` failure as an error on the
/// `LiteLLM` logger before it raises it. Only the failure's own message is logged after "Error:".
fn credentials_failed(error: Error) -> Error {
    let message = litellm_error::find::<GoogleLoginFailure>(&error)
        .map_or(UNDESCRIBED_FAILURE, |failure| failure.message.as_str());
    tracing::error!(
        "Failed to load vertex credentials. Check to see if credentials containing partial/invalid information. Error: {message}"
    );
    error
}

impl VertexLoginService {
    pub fn new(environment: Arc<dyn GoogleEnvironment>) -> Self {
        Self {
            gcp: GcpAuth::default(),
            environment,
            aws: AwsAuthService::default(),
            earlier_key_files: moka::future::Cache::builder()
                .max_capacity(EARLIER_KEY_FILES)
                .build(),
        }
    }

    /// Uses `aws` for an external account's explicit `aws_*` keys, as Bedrock uses it, so an
    /// `aws_web_identity_token` reference resolves through the same web identity tokens.
    pub fn with_aws(self, aws: AwsAuthService) -> Self {
        Self { aws, ..self }
    }

    /// Python's `_ensure_access_token` for `vertex_ai`: `credentials` and `project` as
    /// `safe_get_vertex_ai_credentials` and `safe_get_vertex_ai_project` resolve them.
    pub async fn login(
        &self,
        credentials: Option<String>,
        project: Option<String>,
    ) -> Result<VertexLogin, Error> {
        let adc = match &credentials {
            Some(_) => None,
            None => adc::locate(&*self.environment),
        };
        let key = match (&credentials, &adc) {
            (Some(text), _) => CredentialCacheKey::versioned(text),
            (None, Some(file)) => CredentialCacheKey::versioned(&file.path),
            (None, None) => CredentialCacheKey::metadata(),
        };
        // Python's `load_auth` loads, mints the first token and resolves the project, and logs
        // any failure of those; a call served cached credentials ran no `load_auth`.
        let loaded = AtomicBool::new(false);
        let loaded_source = self
            .gcp
            .load_keyed(key.clone(), async {
                loaded.store(true, Ordering::Relaxed);
                let login = match credentials {
                    Some(text) => self.explicit(text).await,
                    None => adc::default_credentials(&self.environment, adc).await,
                };
                login
                    .map(|login| Arc::new(login) as Arc<dyn GcpTokenSource>)
                    .map_err(Error::from)
            })
            .await;
        let source = self.or_earlier_key_file(&key, loaded_source).await?;
        let token = match tokio::time::timeout(
            TOKEN_TIMEOUT,
            self.gcp.token_evicting(&key, &*source),
        )
        .await
        {
            Err(_) => Err(GoogleLoginFailure::new(
                PythonException::Transport,
                "The token request did not finish in time.",
            )),
            Ok(token) => {
                token.map_err(|error| GoogleLoginFailure::token(&error, &*self.environment))
            }
        }
        .map_err(|failure| match loaded.load(Ordering::Relaxed) {
            true => credentials_failed(failure.into()),
            false => failure.into(),
        })?;
        let project_id = project
            .or(source.project_id().await.ok())
            .filter(|project| !project.is_empty())
            .ok_or_else(|| {
                credentials_failed(value_failure("Could not resolve project_id").into())
            })?;
        Ok(VertexLogin { token, project_id })
    }

    /// Python's `_load_auth_or_earlier_key_file`: credentials a key file loaded are kept for its
    /// path, and a rewrite of that file that does not load is logged and served the credentials
    /// the path loaded last. Credentials that are not a key file, or a path that never loaded,
    /// fail as they are.
    async fn or_earlier_key_file(
        &self,
        key: &CredentialCacheKey,
        loaded: Result<Arc<dyn GcpTokenSource>, Error>,
    ) -> Result<Arc<dyn GcpTokenSource>, Error> {
        let path = match key {
            CredentialCacheKey::KeyFile { path, .. } => Some(path),
            _ => None,
        };
        match (loaded, path) {
            (Ok(source), Some(path)) => {
                self.earlier_key_files
                    .insert(path.clone(), source.clone())
                    .await;
                Ok(source)
            }
            (Ok(source), None) => Ok(source),
            (Err(error), Some(path)) => {
                let error = credentials_failed(error);
                self.earlier_key_files.get(path).await.ok_or(error)
            }
            (Err(error), None) => Err(credentials_failed(error)),
        }
    }

    /// `_credentials_from_json`: an authorized user or an external account by type, a service
    /// account for any other type.
    fn explicit_source(&self, credentials: &CredentialsJson) -> Result<Login, GoogleLoginFailure> {
        let built = match &credentials.credentials {
            ExplicitCredentials::ExternalAccount(info) if uses_explicit_aws(info) => {
                explicit_aws_login(info, &self.environment, &self.aws)
            }
            ExplicitCredentials::ExternalAccount(info) => {
                builders::external_account(info, &self.environment, true)
            }
            ExplicitCredentials::AuthorizedUser(user) => builders::build(
                SdkCredentials::AuthorizedUser {
                    json: user.sdk.clone(),
                    token_uri: self.environment.google_token_endpoint(),
                },
                PythonException::Value,
            ),
            ExplicitCredentials::ServiceAccount(account) => {
                builders::service_account(account, PythonException::Malformed)
            }
        };
        let source = built.map_err(|failure| match &credentials.path {
            Some(path) => GoogleLoginFailure::credentials_file(UnusableFile::Invalid {
                path: path.clone(),
                problem: format!("holds credentials that cannot be used: {}", failure.message),
            }),
            None => failure,
        })?;
        Ok(Login {
            source,
            project: credentials.credentials.project(),
            signs: matches!(
                credentials.credentials,
                ExplicitCredentials::ServiceAccount(_)
            ),
        })
    }

    async fn explicit(&self, text: String) -> Result<Login, GoogleLoginFailure> {
        // The source may be a file to read, which must not block the runtime. The blocking
        // thread logs the file's path and problem through this call's logger and span.
        let dispatch = tracing::dispatcher::get_default(Clone::clone);
        let span = tracing::Span::current();
        let json = tokio::task::spawn_blocking(move || {
            tracing::dispatcher::with_default(&dispatch, || {
                span.in_scope(|| load_vertex_credentials_source(&text))
            })
        })
        .await
        .map_err(|_| value_failure("The `vertex_credentials` file read did not finish"))??;
        self.explicit_source(&json)
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{BTreeMap, VecDeque},
        path::Path,
        sync::Mutex,
    };

    use axum::{
        Json, Router,
        extract::State,
        http::{HeaderMap, StatusCode, Uri},
        response::{IntoResponse, Response},
    };
    use base64::{
        Engine,
        engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD},
    };
    use rstest::{fixture, rstest};
    use serde_json::{Value, json};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::*;
    use crate::{
        CredentialSource, VertexAuth, VertexConfig,
        google::{EnvironmentFuture, OnGce, environment},
    };
    use litellm_auth_types::ErrorDetail;

    const PRIVATE_KEY: &str = "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDGRPkmZkzek9X0\nGGxUpS00yBkTR9RHh1yXLK0SIoIT9SgyZaP5+gETM7Mo3oT4/q1EUnx+9WgvqZPn\nkgMP+/P2vNc07tgYn+DZqkn+M4ys2sGZnK+VQTpcw2AE+4zaSlJJztpGxyMkDAcg\ntHJeKHc0aWxw52d1K+SLzLeIz9Niy3uL5t7iZlFpUDA4LAfTCXlzYW+f+zcXXbof\nTyHRlfl6kvYoOa7ss+9YVXVhp4GOfhLOT4tsYfO3gbNSVrkxzdF6iLG2Wqh9GHMD\n7TdI3aSUXHj7qpwRx6PBY8eH74oHlUlSpi6BHgDGeTK/HdxbrI7WK6hT+9FIP55h\n1Tbk4twjAgMBAAECggEADPgIaNyalplXW2B/MlVp//o+uaYiWo0EhdbxEy7/dzK7\nO5Phn/S2CuPmcJWHLFcfgK9fxiYwET4N/XQgwydMRycFcXu8b6mCWa24qh03f6I0\nTuOS3WtA2ArnkvipUMLS+YT2sVga3yOL3wKVBvFHd6QvT0tNPYeh3j5SNJ7ASV8m\nH2RgRpz9LmOr6CCgOmr/52ldidQiBW3GtqEo8WilF7z+q9tQ6KDFZmcXU+4T2Xzs\nR5ZPIIkZni4X5vDhw7F6NSrbNMbUREDtVStdcwRqmMLoMs9DI6SxEKRnjsV0HxRm\nL0SIs+AiptJPIbVVzWHkMI+L/8J9qiTjl4nozWlZAQKBgQDn0fshqf6yD2SSUf7h\nmpJbkcyw54szZPfJRLftyo0KvKCIGjys+I0+cBsCHPafbOHB70aV9/7pzil3a1uw\nU0C7WOYeTdw9lxm5TVnaAYPJl/iqCfTm+eZkTPnDZRDOeJimoheLN2UB52yKx+78\nXl1AYn/VsnlWd+Kt3sfcng6sQQKBgQDa8yAAHgeDJ1LvfHebzZeJU1QAYw9mO/QJ\nrCMvMTJY2azGJdo61SohDzsWYaFMT/QdlCMYu8W7V2keDZVaJbV2PTulp/XQyOB2\nCOW2899644FMtCkcIv9gDYoVt/xY64TcX5uofq2wdr96pnzlarU8MueUR1qM+mU+\ngJLi1UJ/YwKBgAWihAEweaFoYGByH8jqEe6p35jZN/f6108MHZBpIP6G0TwdryXh\nMMW4wd9ORuxfICmvoD6a9ZZHTZi6glMJ4wPqyEUqTVUt7BEmAS4cZVCsIf44GbzX\njDMZ7+RA+6SZJ9S9EedvZEQklO3WqEd4H7ol6seN/AcPCPKC/MavDNpBAoGBANVc\nRrrWiIuOLouhxwzU8+6mfR1DayY9Fjq2vDKoxl8+/9a33EmooD1HElVlHkG3SDPq\nv6jCkpVVQOJAamJGjOaEFTQA0NJ8bf+X/AKk9ApU2RNeHRGGKUX4667uc8JTAE2f\n/ZfcmnVzwnRK+Tq3GZVoJbOhNORcvXyC3qCdIQfxAoGBAMipsn9l6sqMgNddwtzp\n2n7CuvycTgNQgS+wzy2e3JgQaJYlXTe8J5IVwRbq3MbICm+8gsfMc4cBCvRtmWOZ\nA9P2N1liNx1TYJ2D31nC1FgAt5z+X8ODDqgZjCXvCSVS+AbA0ivkQtzXm7zwLVD7\nKR6kOMrEoPsqjSaI14o9qNFx\n-----END PRIVATE KEY-----\n";
    /// [`PRIVATE_KEY`] in PKCS#1 `RSA PRIVATE KEY` form.
    const PKCS1_PRIVATE_KEY: &str = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEAxkT5JmZM3pPV9BhsVKUtNMgZE0fUR4dclyytEiKCE/UoMmWj\n+foBEzOzKN6E+P6tRFJ8fvVoL6mT55IDD/vz9rzXNO7YGJ/g2apJ/jOMrNrBmZyv\nlUE6XMNgBPuM2kpSSc7aRscjJAwHILRyXih3NGlscOdndSvki8y3iM/TYst7i+be\n4mZRaVAwOCwH0wl5c2Fvn/s3F126H08h0ZX5epL2KDmu7LPvWFV1YaeBjn4Szk+L\nbGHzt4GzUla5Mc3ReoixtlqofRhzA+03SN2klFx4+6qcEcejwWPHh++KB5VJUqYu\ngR4Axnkyvx3cW6yO1iuoU/vRSD+eYdU25OLcIwIDAQABAoIBAAz4CGjcmpaZV1tg\nfzJVaf/6PrmmIlqNBIXW8RMu/3cyuzuT4Z/0tgrj5nCVhyxXH4CvX8YmMBE+Df10\nIMMnTEcnBXF7vG+pglmtuKodN3+iNE7jkt1rQNgK55L4qVDC0vmE9rFYGt8ji98C\nlQbxR3ekL09LTT2Hod4+UjSewElfJh9kYEac/S5jq+ggoDpq/+dpXYnUIgVtxrah\nKPFopRe8/qvbUOigxWZnF1PuE9l87EeWTyCJGZ4uF+bw4cOxejUq2zTG1ERA7VUr\nXXMEapjC6DLPQyOksRCkZ47FdB8UZi9EiLPgIqbSTyG1Vc1h5DCPi//Cfaok45eJ\n6M1pWQECgYEA59H7Ian+sg9kklH+4ZqSW5HMsOeLM2T3yUS37cqNCrygiBo8rPiN\nPnAbAhz2n2zhwe9Glff+6c4pd2tbsFNAu1jmHk3cPZcZuU1Z2gGDyZf4qgn05vnm\nZEz5w2UQzniYpqIXizdlAedsisfu/F5dQGJ/1bJ5Vnfird7H3J4OrEECgYEA2vMg\nAB4HgydS73x3m82XiVNUAGMPZjv0CawjLzEyWNmsxiXaOtUqIQ87FmGhTE/0HZQj\nGLvFu1dpHg2VWiW1dj07paf10MjgdgjltvPfeuOBTLQpHCL/YA2KFbf8WOuE3F+b\nqH6tsHa/eqZ85Wq1PDLnlEdajPplPoCS4tVCf2MCgYAFooQBMHmhaGBgch/I6hHu\nqd+Y2Tf3+tdPDB2QaSD+htE8Ha8l4TDFuMHfTkbsXyApr6A+mvWWR02YuoJTCeMD\n6shFKk1VLewRJgEuHGVQrCH+OBm814wzGe/kQPukmSfUvRHnb2REJJTt1qhHeB+6\nJerHjfwHDwjygvzGrwzaQQKBgQDVXEa61oiLji6LoccM1PPupn0dQ2smPRY6trwy\nqMZfPv/Wt9xJqKA9RxJVZR5Bt0gz6r+owpKVVUDiQGpiRozmhBU0ANDSfG3/l/wC\npPQKVNkTXh0RhilF+Ouu7nPCUwBNn/2X3Jp1c8J0Svk6txmVaCWzoTTkXL18gt6g\nnSEH8QKBgQDIqbJ/ZerKjIDXXcLc6dp+wrr8nE4DUIEvsM8tntyYEGiWJV03vCeS\nFcEW6tzGyApvvILHzHOHAQr0bZljmQPT9jdZYjcdU2Cdg99ZwtRYALec/l/Dgw6o\nGYwl7wklUvgGwNIr5ELc15u88C1Q+ykepDjKxKD7Ko0miNeKPajRcQ==\n-----END RSA PRIVATE KEY-----\n";
    const CLIENT_EMAIL: &str = "vertex@key-project.iam.gserviceaccount.com";
    const STS_JWT: &str = "urn:ietf:params:oauth:token-type:jwt";
    const POOL_AUDIENCE: &str =
        "//iam.googleapis.com/projects/123456/locations/global/workloadIdentityPools/p/providers/q";

    /// A request one of Google's endpoints played by [`Google`] was sent.
    #[derive(Clone, Debug)]
    struct Seen {
        path: String,
        authorization: Option<String>,
        body: String,
    }

    #[derive(Clone, Default)]
    struct Recorded {
        seen: Arc<Mutex<Vec<Seen>>>,
        answers: Arc<Mutex<VecDeque<(u16, Value)>>>,
    }

    fn granted() -> Value {
        json!({
            "access_token": "granted", "expires_in": 3600, "token_type": "Bearer",
            "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
        })
    }

    async fn answer(
        State(recorded): State<Recorded>,
        uri: Uri,
        headers: HeaderMap,
        body: String,
    ) -> Response {
        recorded.seen.lock().unwrap().push(Seen {
            path: uri.path().to_owned(),
            authorization: headers
                .get("authorization")
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned),
            body,
        });
        match uri.path() {
            path if path.ends_with("refused@p.iam.gserviceaccount.com:generateAccessToken") => {
                (StatusCode::FORBIDDEN, Json(json!({"error": "denied"}))).into_response()
            }
            path if path.ends_with(":generateAccessToken") => {
                let expires = time::OffsetDateTime::now_utc() + time::Duration::hours(1);
                let expire_time = expires
                    .format(&time::format_description::well_known::Rfc3339)
                    .unwrap();
                Json(json!({"accessToken": "impersonated", "expireTime": expire_time}))
                    .into_response()
            }
            "/subject" => "url-subject".into_response(),
            "/aws/region" => "us-east-1a".into_response(),
            "/aws/credentials" => "role".into_response(),
            "/aws/credentials/role" => Json(json!({
                "AccessKeyId": "AKIDMETADATA", "SecretAccessKey": "metadata-secret", "Token": "session",
            }))
            .into_response(),
            _ => {
                let (status, body) = recorded
                    .answers
                    .lock()
                    .unwrap()
                    .pop_front()
                    .unwrap_or_else(|| (200, granted()));
                (StatusCode::from_u16(status).unwrap(), Json(body)).into_response()
            }
        }
    }

    /// Google's token, STS, IAM and AWS metadata endpoints played by one local server, which
    /// answers a token request with the next queued answer or else grants `granted`.
    struct Google {
        url: String,
        recorded: Recorded,
        server: tokio::task::JoinHandle<()>,
    }

    impl Drop for Google {
        fn drop(&mut self) {
            self.server.abort();
        }
    }

    impl Google {
        fn token_url(&self) -> String {
            format!("{}/token", self.url)
        }

        fn queue(&self, status: u16, body: Value) {
            self.recorded
                .answers
                .lock()
                .unwrap()
                .push_back((status, body));
        }

        fn sent_to(&self, path: &str) -> Vec<Seen> {
            self.recorded
                .seen
                .lock()
                .unwrap()
                .iter()
                .filter(|seen| seen.path == path)
                .cloned()
                .collect()
        }
    }

    #[fixture]
    async fn google() -> Google {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let recorded = Recorded::default();
        let app = Router::new().fallback(answer).with_state(recorded.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Google {
            url: format!("http://{address}"),
            recorded,
            server,
        }
    }

    /// The machine a test login runs on: its environment, gcloud, the metadata server and Cloud
    /// Resource Manager, with litellm's own requests made under `env` alone.
    #[derive(Default)]
    struct FakeEnvironment {
        token_endpoint: String,
        env: BTreeMap<&'static str, String>,
        gcloud_project: Option<&'static str>,
        on_gce: bool,
        gce_project: Option<&'static str>,
        resource_manager_project: Option<&'static str>,
        resource_manager_urls: Mutex<Vec<String>>,
    }

    impl GoogleEnvironment for FakeEnvironment {
        fn os_env(&self, name: &str) -> Option<String> {
            self.env.get(name).cloned()
        }

        fn google_token_endpoint(&self) -> String {
            self.token_endpoint.clone()
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

        fn gce_credentials(&self) -> Result<Arc<dyn GcpTokenSource>, GoogleLoginFailure> {
            OnGce {
                project: "unused",
                token: "metadata-token",
            }
            .gce_credentials()
        }

        fn resource_manager_project(
            &self,
            url: String,
            _token: SecretValue,
        ) -> EnvironmentFuture<'_, Result<Option<String>, GoogleLoginFailure>> {
            self.resource_manager_urls.lock().unwrap().push(url);
            Box::pin(async { Ok(self.resource_manager_project.map(str::to_string)) })
        }

        fn http_client(
            &self,
            url: &str,
            identity: Option<litellm_http::ClientIdentity>,
        ) -> Result<litellm_http::Client, GoogleLoginFailure> {
            let env: Vec<(String, String)> = self
                .env
                .iter()
                .map(|(name, value)| ((*name).to_owned(), value.clone()))
                .collect();
            environment::requests_client(&env, url, identity)
        }
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

    /// Google's OAuth token endpoint played by `google`.
    fn at(google: &Google) -> FakeEnvironment {
        FakeEnvironment {
            token_endpoint: google.token_url(),
            ..FakeEnvironment::default()
        }
    }

    fn auth(environment: FakeEnvironment) -> VertexLoginService {
        VertexLoginService::new(Arc::new(environment))
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

    fn merged(base: Value, extra: Value) -> Value {
        Value::Object(
            base.as_object()
                .unwrap()
                .clone()
                .into_iter()
                .chain(extra.as_object().unwrap().clone())
                .collect(),
        )
    }

    fn authorized_user(extra: Value) -> Value {
        merged(
            json!({
                "type": "authorized_user", "client_id": "client", "client_secret": "secret",
                "refresh_token": "refresh", "token_uri": "http://127.0.0.1:9/token",
            }),
            extra,
        )
    }

    fn service_account(extra: Value) -> Value {
        merged(
            json!({
                "type": "service_account", "client_email": CLIENT_EMAIL, "private_key": PRIVATE_KEY,
                "private_key_id": "key-1", "project_id": "key-project",
                "token_uri": "https://oauth2.googleapis.com/token",
            }),
            extra,
        )
    }

    fn external_account(google: &Google, source: Value) -> Value {
        json!({
            "type": "external_account", "audience": POOL_AUDIENCE, "subject_token_type": STS_JWT,
            "token_url": google.token_url(), "credential_source": source,
        })
    }

    /// A JWT's header and claims.
    fn decoded(token: &str) -> (Value, Value) {
        let part = |index: usize| -> Value {
            serde_json::from_slice(
                &URL_SAFE_NO_PAD
                    .decode(token.split('.').nth(index).unwrap())
                    .unwrap(),
            )
            .unwrap()
        };
        (part(0), part(1))
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
    #[case::with_key_id(json!({}), "key-1", "key-project")]
    #[case::without_key_id_or_project(json!({"private_key_id": null, "project_id": null}), "", "configured")]
    #[tokio::test]
    async fn a_service_account_signs_its_own_token(
        #[future] google: Google,
        #[case] extra: Value,
        #[case] key_id: &str,
        #[case] project: &str,
    ) {
        let google = google.await;
        let login = auth(at(&google))
            .login(
                Some(
                    service_account(merged(extra, json!({"token_uri": google.token_url()})))
                        .to_string(),
                ),
                (project == "configured").then(|| "configured".into()),
            )
            .await
            .unwrap();
        let (header, claims) = decoded(login.token.expose());
        assert_eq!(header["kid"], key_id);
        assert_eq!(header["alg"], "RS256");
        assert_eq!(claims["iss"], CLIENT_EMAIL);
        assert_eq!(
            claims["scope"],
            "https://www.googleapis.com/auth/cloud-platform"
        );
        assert_eq!(login.project_id, project);
        assert!(google.sent_to("/token").is_empty());
    }

    #[rstest]
    #[case::pkcs8(PRIVATE_KEY)]
    #[case::pkcs1(PKCS1_PRIVATE_KEY)]
    #[tokio::test]
    async fn a_service_account_token_verifies_against_the_key_s_public_half(
        #[future] google: Google,
        #[case] private_key: &str,
    ) {
        use rsa::{
            RsaPrivateKey,
            pkcs1v15::{Signature, VerifyingKey},
            pkcs8::DecodePrivateKey,
            sha2::Sha256,
            signature::Verifier,
        };

        let google = google.await;
        let login = auth(at(&google))
            .login(
                Some(service_account(json!({"private_key": private_key})).to_string()),
                None,
            )
            .await
            .unwrap();
        let token = login.token.expose();
        let (signed, signature) = token.rsplit_once('.').unwrap();
        let public = VerifyingKey::<Sha256>::new(
            RsaPrivateKey::from_pkcs8_pem(PRIVATE_KEY)
                .unwrap()
                .to_public_key(),
        );
        let signature =
            Signature::try_from(URL_SAFE_NO_PAD.decode(signature).unwrap().as_slice()).unwrap();
        assert!(public.verify(signed.as_bytes(), &signature).is_ok());
    }

    #[rstest]
    #[tokio::test]
    async fn an_authorized_user_refreshes_at_google_whatever_token_uri_it_names(
        #[future] google: Google,
    ) {
        let google = google.await;
        let login = auth(at(&google))
            .login(
                Some(authorized_user(json!({"quota_project_id": "quota"})).to_string()),
                None,
            )
            .await
            .unwrap();
        assert_eq!(
            (login.token.expose(), login.project_id.as_str()),
            ("granted", "quota")
        );
        assert!(google.sent_to("/token")[0].body.contains("refresh"));
    }

    fn file_source(google: &Google, directory: &Path) -> Value {
        let subject = directory.join("subject");
        std::fs::write(&subject, "file-subject").unwrap();
        external_account(google, json!({"file": subject}))
    }

    fn url_source(google: &Google, _directory: &Path) -> Value {
        external_account(google, json!({"url": format!("{}/subject", google.url)}))
    }

    fn aws_metadata_source(google: &Google, _directory: &Path) -> Value {
        merged(
            external_account(
                google,
                json!({
                    "environment_id": "aws1",
                    "region_url": format!("{}/aws/region", google.url),
                    "url": format!("{}/aws/credentials", google.url),
                    "regional_cred_verification_url": "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15",
                }),
            ),
            json!({"subject_token_type": "urn:ietf:params:aws:token-type:aws4_request"}),
        )
    }

    fn impersonating_source(google: &Google, directory: &Path) -> Value {
        merged(
            file_source(google, directory),
            json!({"service_account_impersonation_url": format!(
                "{}/v1/projects/-/serviceAccounts/sa@p.iam.gserviceaccount.com:generateAccessToken",
                google.url
            )}),
        )
    }

    fn explicit_aws(google: &Google, _directory: &Path) -> Value {
        merged(
            aws_metadata_source(google, _directory),
            json!({
                "aws_access_key_id": "AKIDEXPLICIT", "aws_secret_access_key": "explicit-secret",
                "aws_region_name": "us-east-1",
            }),
        )
    }

    fn explicit_aws_impersonating(google: &Google, directory: &Path) -> Value {
        merged(
            explicit_aws(google, directory),
            json!({"service_account_impersonation_url": format!(
                "{}/v1/projects/-/serviceAccounts/sa@p.iam.gserviceaccount.com:generateAccessToken",
                google.url
            )}),
        )
    }

    #[rstest]
    #[case::file(file_source, "granted", "subject_token=file-subject")]
    #[case::url(url_source, "granted", "subject_token=url-subject")]
    #[cfg_attr(
        not(feature = "fips"),
        case::aws_metadata(aws_metadata_source, "granted", "subject_token=%257B")
    )]
    #[case::impersonated(impersonating_source, "impersonated", "subject_token=file-subject")]
    #[case::explicit_aws(explicit_aws, "granted", "subject_token=%257B%2522headers%2522")]
    #[case::explicit_aws_impersonated(
        explicit_aws_impersonating,
        "impersonated",
        "subject_token=%257B%2522headers%2522"
    )]
    #[tokio::test]
    async fn an_external_account_exchanges_its_subject_token_at_sts(
        #[future] google: Google,
        #[case] account: fn(&Google, &Path) -> Value,
        #[case] token: &str,
        #[case] subject: &str,
    ) {
        let google = google.await;
        let directory = tempfile::tempdir().unwrap();
        let login = auth(at(&google))
            .login(
                Some(account(&google, directory.path()).to_string()),
                Some("p".into()),
            )
            .await
            .unwrap();
        assert_eq!(login.token.expose(), token);
        let exchanges = google.sent_to("/token");
        assert!(exchanges[0].body.contains(subject), "{exchanges:?}");
    }

    #[cfg(feature = "fips")]
    #[rstest]
    #[case::vertex_credentials(explicit(|google| aws_metadata_source(google, Path::new("/"))), PythonException::Malformed)]
    #[case::application_default_file(Box::new(|google: &Google, directory: &Path| adc_file(aws_metadata_source(google, directory))(google, directory)), PythonException::DefaultCredentials)]
    #[case::vertex_credentials_with_a_repeated_key(Box::new(|google: &Google, directory: &Path| (Some(with_a_repeated_environment_id(aws_metadata_source(google, directory))), at(google))), PythonException::Malformed)]
    #[case::application_default_file_with_a_repeated_key(Box::new(|google: &Google, directory: &Path| adc_text(with_a_repeated_environment_id(aws_metadata_source(google, directory)))(google, directory)), PythonException::DefaultCredentials)]
    #[tokio::test]
    async fn the_fips_build_refuses_an_aws_source_google_s_sdk_would_sign(
        #[future] google: Google,
        #[case] setup: Setup,
        #[case] exception: PythonException,
    ) {
        let google = google.await;
        let directory = tempfile::tempdir().unwrap();
        let (credentials, environment) = setup(&google, directory.path());

        let refused = failure(
            auth(environment)
                .login(credentials, Some("p".into()))
                .await
                .unwrap_err(),
        );

        assert_eq!(
            (refused.exception, refused.message.as_str()),
            (exception, crate::google::builders::AWS_SOURCE_IN_FIPS)
        );
        assert!(google.sent_to("/token").is_empty());
    }

    #[rstest]
    #[tokio::test]
    async fn an_allowed_executable_supplies_the_subject_token(#[future] google: Google) {
        let google = google.await;
        let directory = tempfile::tempdir().unwrap();
        let script = directory.path().join("subject.sh");
        let expires = time::OffsetDateTime::now_utc().unix_timestamp() + 3600;
        std::fs::write(
            &script,
            format!(
                "#!/bin/sh\necho '{{\"version\": 1, \"success\": true, \"token_type\": \"{STS_JWT}\", \"id_token\": \"executable-subject\", \"expiration_time\": {expires}}}'\n"
            ),
        )
        .unwrap();
        std::fs::set_permissions(
            &script,
            <std::fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o755),
        )
        .unwrap();
        let configured = external_account(
            &google,
            json!({"executable": {"command": script, "timeout_millis": 5000}}),
        );
        let login = auth(environment(&[(
            "GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES",
            "1",
        )]))
        .login(Some(configured.to_string()), Some("p".into()))
        .await
        .unwrap();
        assert_eq!(login.token.expose(), "granted");
        assert!(
            google.sent_to("/token")[0]
                .body
                .contains("subject_token=executable-subject")
        );
    }

    #[rstest]
    #[case::unset(&[])]
    #[case::zero(&[("GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES", "0")])]
    #[case::true_word(&[("GOOGLE_EXTERNAL_ACCOUNT_ALLOW_EXECUTABLES", "true")])]
    #[tokio::test]
    async fn an_executable_runs_only_when_allowed_with_1(
        #[future] google: Google,
        #[case] env: &[(&'static str, &str)],
    ) {
        let google = google.await;
        let configured = external_account(&google, json!({"executable": {"command": "/bin/true"}}));

        let error = auth(environment(env))
            .login(Some(configured.to_string()), Some("p".into()))
            .await
            .unwrap_err();

        assert_eq!(failure(error).exception, PythonException::Malformed);
    }

    #[rstest]
    #[case::default_lifetime(json!({}), "3600s")]
    #[case::configured_lifetime(json!({"service_account_impersonation": {"token_lifetime_seconds": 1800}}), "1800s")]
    #[tokio::test]
    async fn a_custom_impersonation_url_is_called_with_the_sts_token(
        #[future] google: Google,
        #[case] extra: Value,
        #[case] lifetime: &str,
    ) {
        let google = google.await;
        let directory = tempfile::tempdir().unwrap();
        let configured = merged(impersonating_source(&google, directory.path()), extra);

        let login = auth(at(&google))
            .login(Some(configured.to_string()), Some("p".into()))
            .await
            .unwrap();

        let impersonations = google.sent_to(
            "/v1/projects/-/serviceAccounts/sa@p.iam.gserviceaccount.com:generateAccessToken",
        );
        let body: Value = serde_json::from_str(&impersonations[0].body).unwrap();
        assert_eq!(
            (
                login.token.expose(),
                impersonations[0].authorization.as_deref(),
                body
            ),
            (
                "impersonated",
                Some("Bearer granted"),
                json!({"delegates": null, "scope": ["https://www.googleapis.com/auth/cloud-platform"], "lifetime": lifetime})
            )
        );
    }

    #[rstest]
    #[case::configured_project_wins(json!({"quota_project_id": "quota"}), Some("configured"), Ok("configured"))]
    #[case::authorized_user_quota_project(json!({"quota_project_id": "quota"}), None, Ok("quota"))]
    #[case::no_quota_project(json!({}), None, Err("Could not resolve project_id"))]
    #[case::null_quota_project(json!({"quota_project_id": null}), None, Err("Could not resolve project_id"))]
    #[case::empty_configured_project(json!({"quota_project_id": "quota"}), Some(""), Err("Could not resolve project_id"))]
    #[tokio::test]
    async fn explicit_credentials_name_the_project_as_load_auth_does(
        #[future] google: Google,
        #[case] extra: Value,
        #[case] configured: Option<&str>,
        #[case] expected: Result<&str, &str>,
    ) {
        let google = google.await;
        let login = auth(at(&google))
            .login(
                Some(authorized_user(extra).to_string()),
                configured.map(str::to_string),
            )
            .await;
        assert_eq!(
            login
                .map(|login| login.project_id)
                .map_err(|error| failure(error).message),
            expected.map(str::to_owned).map_err(str::to_owned)
        );
        assert_eq!(google.sent_to("/token").len(), 1);
    }

    #[rstest]
    #[case::credentials_file_project(&[], "file-project")]
    #[case::google_cloud_project_first(&[("GOOGLE_CLOUD_PROJECT", "env-project")], "env-project")]
    #[case::legacy_project_name(&[("GCLOUD_PROJECT", "legacy-project")], "legacy-project")]
    #[case::empty_project_variable_is_skipped(&[("GOOGLE_CLOUD_PROJECT", "")], "file-project")]
    #[tokio::test]
    async fn the_gcloud_credentials_file_names_the_project_as_google_auth_does(
        #[future] google: Google,
        #[case] env: &[(&'static str, &str)],
        #[case] expected: &str,
    ) {
        let google = google.await;
        let gcloud = tempfile::tempdir().unwrap();
        std::fs::write(
            gcloud.path().join("application_default_credentials.json"),
            authorized_user(json!({"quota_project_id": "quota"})).to_string(),
        )
        .unwrap();
        let fake = FakeEnvironment {
            gcloud_project: Some("file-project"),
            token_endpoint: google.token_url(),
            ..environment(&[env, &[("CLOUDSDK_CONFIG", gcloud.path().to_str().unwrap())]].concat())
        };
        let login = auth(fake).login(None, None).await.unwrap();
        assert_eq!(
            (login.project_id.as_str(), login.token.expose()),
            (expected, "granted")
        );
    }

    #[rstest]
    #[tokio::test]
    async fn an_authorized_user_named_by_google_application_credentials_carries_no_project(
        #[future] google: Google,
    ) {
        let google = google.await;
        let file = key_file(&authorized_user(json!({"quota_project_id": "quota"})));
        let fake = FakeEnvironment {
            gcloud_project: Some("gcloud-project"),
            token_endpoint: google.token_url(),
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
        assert_eq!(
            (login.project_id.as_str(), login.token.expose()),
            ("gce-project", "metadata-token")
        );
    }

    #[rstest]
    #[case::project_found(&[], Some("crm-project"), Ok("crm-project"), 1)]
    #[case::project_not_found(&[], None, Err("Could not resolve project_id"), 1)]
    #[case::environment_project_skips_it(&[("GOOGLE_CLOUD_PROJECT", "env-project")], Some("crm-project"), Ok("env-project"), 0)]
    #[tokio::test]
    async fn an_external_account_default_asks_resource_manager_for_its_project(
        #[future] google: Google,
        #[case] env: &[(&'static str, &str)],
        #[case] answered: Option<&'static str>,
        #[case] expected: Result<&str, &str>,
        #[case] asked: usize,
    ) {
        let google = google.await;
        let directory = tempfile::tempdir().unwrap();
        let file = key_file(&file_source(&google, directory.path()));
        let fake = Arc::new(FakeEnvironment {
            resource_manager_project: answered,
            ..environment(
                &[
                    env,
                    &[(
                        "GOOGLE_APPLICATION_CREDENTIALS",
                        file.path().to_str().unwrap(),
                    )],
                ]
                .concat(),
            )
        });
        let login = VertexLoginService::new(fake.clone())
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
            vec![
                "https://cloudresourcemanager.googleapis.com/v1/projects/123456".to_owned();
                asked
            ]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn an_application_default_impersonated_service_account_logs_in(#[future] google: Google) {
        let google = google.await;
        let file = key_file(&json!({
            "type": "impersonated_service_account",
            "service_account_impersonation_url": format!(
                "{}/v1/projects/-/serviceAccounts/sa@p.iam.gserviceaccount.com:generateAccessToken",
                google.url
            ),
            "source_credentials": authorized_user(json!({"token_uri": google.token_url()})),
        }));
        let login = auth(environment(&[(
            "GOOGLE_APPLICATION_CREDENTIALS",
            file.path().to_str().unwrap(),
        )]))
        .login(None, Some("p".into()))
        .await
        .unwrap();
        assert_eq!(login.token.expose(), "impersonated");
    }

    #[rstest]
    #[tokio::test]
    async fn an_application_default_external_account_authorized_user_refreshes_at_its_token_url(
        #[future] google: Google,
    ) {
        let google = google.await;
        let file = key_file(&json!({
            "type": "external_account_authorized_user",
            "audience": "//iam.googleapis.com/locations/global/workforcePools/p/providers/q",
            "refresh_token": "workforce-refresh", "token_url": google.token_url(),
            "client_id": "c", "client_secret": "s",
        }));
        let auth = auth(environment(&[(
            "GOOGLE_APPLICATION_CREDENTIALS",
            file.path().to_str().unwrap(),
        )]));
        let first = auth.login(None, Some("p".into())).await.unwrap();
        let second = auth.login(None, Some("p".into())).await.unwrap();
        assert_eq!(
            (first.token.expose(), second.token.expose()),
            ("granted", "granted")
        );
        let refreshes = google.sent_to("/token");
        assert_eq!(refreshes.len(), 1);
        assert_eq!(
            refreshes[0].authorization.as_deref(),
            Some(format!("Basic {}", STANDARD.encode("c:s")).as_str())
        );
        assert!(
            refreshes[0]
                .body
                .contains("refresh_token=workforce-refresh")
        );
    }

    fn adc_file(contents: Value) -> impl Fn(&Google, &Path) -> (Option<String>, FakeEnvironment) {
        adc_text(contents.to_string())
    }

    fn adc_text(contents: String) -> impl Fn(&Google, &Path) -> (Option<String>, FakeEnvironment) {
        move |_, directory| {
            let path = directory.join("adc.json");
            std::fs::write(&path, &contents).unwrap();
            (
                None,
                environment(&[("GOOGLE_APPLICATION_CREDENTIALS", path.to_str().unwrap())]),
            )
        }
    }

    type Setup = Box<dyn Fn(&Google, &Path) -> (Option<String>, FakeEnvironment)>;

    #[cfg(feature = "fips")]
    fn with_a_repeated_environment_id(account: Value) -> String {
        let text = account.to_string();
        let repeated = text.replacen(
            r#""environment_id":"aws1""#,
            r#""environment_id":"aws1","environment_id":"aws1""#,
            1,
        );
        assert_ne!(repeated, text);
        repeated
    }

    fn explicit(credentials: impl Fn(&Google) -> Value + 'static) -> Setup {
        Box::new(move |google, _| (Some(credentials(google).to_string()), at(google)))
    }

    fn refused(status: u16, body: Value) -> Option<(u16, Value)> {
        Some((status, body))
    }

    #[rstest]
    #[case::no_credentials_anywhere(Box::new(|_: &Google, _: &Path| (None, FakeEnvironment::default())), None, PythonException::DefaultCredentials)]
    #[case::missing_application_default_file(Box::new(|_: &Google, _: &Path| (None, environment(&[("GOOGLE_APPLICATION_CREDENTIALS", "/nonexistent/adc.json")]))), None, PythonException::DefaultCredentials)]
    #[case::application_default_file_not_an_object(Box::new(adc_file(json!("{oops"))), None, PythonException::DefaultCredentials)]
    #[case::application_default_file_of_unknown_type(Box::new(adc_file(json!({"type": "weird"}))), None, PythonException::DefaultCredentials)]
    #[case::application_default_gdch(Box::new(adc_file(json!({"type": "gdch_service_account"}))), None, PythonException::DefaultCredentials)]
    #[case::vertex_credentials_file_missing(Box::new(|google: &Google, _: &Path| (Some("/nonexistent/vertex-key.json".into()), at(google))), None, PythonException::Value)]
    #[case::service_account_without_a_private_key(explicit(|_| json!({"client_email": CLIENT_EMAIL, "token_uri": "https://t"})), None, PythonException::Malformed)]
    #[case::service_account_with_an_unloadable_key(explicit(|_| service_account(json!({"private_key": "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n"}))), None, PythonException::Value)]
    #[case::authorized_user_missing_a_field(explicit(|_| json!({"type": "authorized_user", "client_id": "c", "client_secret": "s"})), None, PythonException::Value)]
    #[case::external_account_without_a_source(explicit(|google| json!({"type": "external_account", "audience": POOL_AUDIENCE, "token_url": google.token_url()})), None, PythonException::InvalidValue)]
    #[case::executable_not_allowed(explicit(|google| external_account(google, json!({"executable": {"command": "/bin/true"}}))), None, PythonException::Malformed)]
    #[case::explicit_aws_without_a_region(explicit(|google| merged(aws_metadata_source(google, Path::new("/")), json!({"aws_access_key_id": "AKID"}))), None, PythonException::Value)]
    #[case::refresh_refused(explicit(|_| authorized_user(json!({}))), refused(400, json!({"error": "invalid_grant"})), PythonException::Refresh)]
    #[case::refresh_server_error(explicit(|_| authorized_user(json!({}))), refused(500, json!({"error": "internal"})), PythonException::Refresh)]
    #[case::refresh_answers_no_token(explicit(|_| authorized_user(json!({}))), refused(200, json!({"expires_in": 3600})), PythonException::Refresh)]
    #[case::custom_impersonation_refused(Box::new(|google: &Google, directory: &Path| (Some(merged(file_source(google, directory), json!({"service_account_impersonation_url": format!("{}/v1/projects/-/serviceAccounts/refused@p.iam.gserviceaccount.com:generateAccessToken", google.url)})).to_string()), at(google))), None, PythonException::Refresh)]
    #[case::custom_impersonation_over_plain_http(Box::new(|google: &Google, directory: &Path| (Some(merged(file_source(google, directory), json!({"service_account_impersonation_url": "http://iamcredentials.example.invalid/v1/projects/-/serviceAccounts/sa@p.iam.gserviceaccount.com:generateAccessToken"})).to_string()), at(google))), None, PythonException::Refresh)]
    #[case::exchange_refused(Box::new(|google: &Google, directory: &Path| (Some(file_source(google, directory).to_string()), at(google))), refused(400, json!({"error": "invalid_grant"})), PythonException::OAuth)]
    #[case::workforce_refresh_refused(Box::new(|google: &Google, directory: &Path| adc_file(json!({"type": "external_account_authorized_user", "refresh_token": "r", "token_url": google.token_url(), "client_id": "c", "client_secret": "s"}))(google, directory)), refused(401, json!({"error": "invalid_client"})), PythonException::OAuth)]
    #[case::workforce_user_that_cannot_load(Box::new(adc_file(json!({"type": "external_account_authorized_user"}))), None, PythonException::DefaultCredentials)]
    #[case::workforce_user_that_cannot_refresh(Box::new(adc_file(json!({"type": "external_account_authorized_user", "token": "file-token"}))), None, PythonException::Refresh)]
    #[case::token_endpoint_unreachable(Box::new(|_: &Google, _: &Path| (Some(authorized_user(json!({})).to_string()), FakeEnvironment { token_endpoint: "http://127.0.0.1:9/token".into(), ..FakeEnvironment::default() })), None, PythonException::Transport)]
    #[tokio::test]
    async fn a_failed_login_is_raised_as_google_auth_s_exception(
        #[future] google: Google,
        #[case] setup: Setup,
        #[case] answer: Option<(u16, Value)>,
        #[case] exception: PythonException,
    ) {
        let google = google.await;
        let directory = tempfile::tempdir().unwrap();
        if let Some((status, body)) = answer {
            google.queue(status, body);
        }
        let (credentials, environment) = setup(&google, directory.path());
        let error = auth(environment)
            .login(credentials, Some("p".into()))
            .await
            .unwrap_err();
        assert_eq!(failure(error).exception, exception);
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
                "The service account key is missing token_uri, private_key."
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
            failure(error).file,
            Some(UnusableFile::Invalid {
                path: file.path().to_str().unwrap().into(),
                problem: "holds credentials that cannot be used: The service account key is missing client_email, token_uri, private_key.".into(),
            })
        );
    }

    #[derive(Clone, Default)]
    struct Errors(Arc<Mutex<Vec<String>>>);

    impl litellm_tracing::Sink for Errors {
        fn enabled(&self, metadata: &litellm_tracing::Metadata<'_>) -> bool {
            *metadata.level() == litellm_tracing::Level::ERROR
        }

        fn emit(&self, record: &litellm_tracing::Record) {
            self.0.lock().unwrap().push(record.message.clone());
        }
    }

    #[rstest]
    #[tokio::test]
    async fn a_credentials_file_read_off_the_runtime_still_logs_its_path_and_problem() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), "not json").unwrap();
        let path = file.path().to_str().unwrap().to_owned();
        let errors = Errors::default();
        let error = litellm_tracing::Logger::new(errors.clone())
            .instrument(
                auth(FakeEnvironment::default()).login(Some(path.clone()), Some("project".into())),
            )
            .await
            .unwrap_err();
        assert!(failure(error).file.is_some());
        let logged = errors.0.lock().unwrap().clone();
        assert!(
            logged.iter().any(|line| line.starts_with(&format!(
                "Vertex: the credentials file at '{path}' is not valid JSON"
            ))),
            "{logged:?}"
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_rewritten_key_file_loads_anew() {
        let file = key_file(&service_account(json!({})));
        let path = file.path().to_str().unwrap().to_owned();
        let auth = auth(FakeEnvironment::default());
        let before = auth.login(Some(path.clone()), None).await.unwrap();
        std::fs::write(
            file.path(),
            service_account(json!({"client_email": "rotated@key-project.iam.gserviceaccount.com"}))
                .to_string(),
        )
        .unwrap();
        let after = auth.login(Some(path), None).await.unwrap();
        assert_eq!(decoded(before.token.expose()).1["iss"], CLIENT_EMAIL);
        assert_eq!(
            decoded(after.token.expose()).1["iss"],
            "rotated@key-project.iam.gserviceaccount.com"
        );
    }

    const BROKEN_KEY_FILE: &str = "{not a key file";
    const ROTATED_EMAIL: &str = "rotated@key-project.iam.gserviceaccount.com";

    fn good_key() -> String {
        service_account(json!({})).to_string()
    }

    fn rotated_key() -> String {
        service_account(json!({"client_email": ROTATED_EMAIL})).to_string()
    }

    /// `auth` logged in with each of `earlier` written in turn to `file`, and then `file` holding
    /// `last`.
    async fn rewritten(auth: &VertexLoginService, file: &Path, earlier: &[String], last: &str) {
        for contents in earlier {
            std::fs::write(file, contents).unwrap();
            let _ = auth.login(Some(file.to_str().unwrap().into()), None).await;
        }
        std::fs::write(file, last).unwrap();
    }

    #[rstest]
    #[case::good_then_broken(vec![good_key()], BROKEN_KEY_FILE.into(), Ok(CLIENT_EMAIL), true)]
    #[case::broken_first(vec![], BROKEN_KEY_FILE.into(), Err(PythonException::Value), true)]
    #[case::broken_then_fixed(vec![good_key(), BROKEN_KEY_FILE.into()], rotated_key(), Ok(ROTATED_EMAIL), false)]
    #[tokio::test]
    async fn a_key_file_rewrite_that_does_not_load_keeps_the_earlier_credentials(
        #[case] earlier: Vec<String>,
        #[case] last: String,
        #[case] expected: Result<&str, PythonException>,
        #[case] logged: bool,
    ) {
        let file = tempfile::NamedTempFile::new().unwrap();
        let auth = auth(FakeEnvironment::default());
        rewritten(&auth, file.path(), &earlier, &last).await;
        let errors = Errors::default();

        let login = litellm_tracing::Logger::new(errors.clone())
            .instrument(auth.login(Some(file.path().to_str().unwrap().into()), None))
            .await;

        assert_eq!(
            login
                .map(|login| decoded(login.token.expose()).1["iss"].clone())
                .map_err(|error| failure(error).exception),
            expected.map(Value::from)
        );
        assert_eq!(
            errors
                .0
                .lock()
                .unwrap()
                .iter()
                .any(|line| line.starts_with("Failed to load vertex credentials.")),
            logged
        );
    }

    #[rstest]
    #[case::kept_while_granted(None, Ok("granted"), 1)]
    #[case::loaded_anew_after_a_refusal(refused(400, json!({"error": "invalid_grant"})), Ok("granted"), 2)]
    #[tokio::test]
    async fn credentials_are_reused_until_google_refuses_them(
        #[future] google: Google,
        #[case] first: Option<(u16, Value)>,
        #[case] second: Result<&str, ()>,
        #[case] requests: usize,
    ) {
        let google = google.await;
        if let Some((status, body)) = first {
            google.queue(status, body);
        }
        let auth = auth(at(&google));
        let credentials = authorized_user(json!({})).to_string();
        let _ = auth
            .login(Some(credentials.clone()), Some("p".into()))
            .await;
        let again = auth.login(Some(credentials), Some("p".into())).await;
        assert_eq!(
            again
                .map(|login| login.token.expose().to_owned())
                .map_err(|_| ()),
            second.map(str::to_owned)
        );
        assert_eq!(google.sent_to("/token").len(), requests);
    }

    #[rstest]
    #[tokio::test]
    async fn the_shared_loader_keeps_main_s_login_for_other_consumers(#[future] google: Google) {
        let google = google.await;
        let credentials =
            authorized_user(json!({"quota_project_id": "quota", "token_uri": google.token_url()}))
                .to_string();
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
            Err(Error::CredentialAcquisition(ErrorDetail::Unresolved {
                subject: "GCP resource project"
            }))
        ));
        let messages = auth(at(&google))
            .login(Some(credentials), None)
            .await
            .unwrap();
        assert_eq!(messages.project_id, "quota");
    }

    #[rstest]
    #[tokio::test]
    async fn the_shared_loader_ignores_explicit_aws_keys(#[future] google: Google) {
        let google = google.await;
        let explicit_aws = merged(
            aws_metadata_source(&google, Path::new("/")),
            json!({"aws_access_key_id": "key"}),
        )
        .to_string();
        let shared = GcpAuth::default()
            .load_provider(CredentialSource::Trusted(SecretValue::new(
                explicit_aws.clone(),
            )))
            .await;
        assert_eq!(
            shared.is_ok(),
            !cfg!(feature = "fips"),
            "{:?}",
            shared.err()
        );
        let error = auth(FakeEnvironment::default())
            .login(Some(explicit_aws), Some("p".into()))
            .await
            .unwrap_err();
        assert_eq!(failure(error).exception, PythonException::Value);
    }

    /// A TLS endpoint on `127.0.0.1` with `server`'s certificate and client verification, which
    /// answers each request with a granted token and keeps its body.
    async fn tls_token_endpoint(server: rustls::ServerConfig) -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!(
            "https://127.0.0.1:{}/token",
            listener.local_addr().unwrap().port()
        );
        let bodies = Arc::new(Mutex::new(Vec::new()));
        let recorded = bodies.clone();
        let acceptor = tokio_rustls::TlsAcceptor::from(Arc::new(server));
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let Ok(mut tls) = acceptor.accept(stream).await else {
                    continue;
                };
                let mut request = Vec::new();
                let mut chunk = [0; 4096];
                while let Ok(read) = tls.read(&mut chunk).await {
                    request.extend_from_slice(&chunk[..read]);
                    let text = String::from_utf8_lossy(&request).into_owned();
                    let Some((head, body)) = text.split_once("\r\n\r\n") else {
                        continue;
                    };
                    let length = head
                        .lines()
                        .find_map(|line| {
                            line.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .map(str::to_owned)
                        })
                        .and_then(|length| length.trim().parse().ok())
                        .unwrap_or(0);
                    if body.len() >= length || read == 0 {
                        recorded.lock().unwrap().push(body.to_owned());
                        break;
                    }
                }
                let body = granted().to_string();
                let _ = tls
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .await;
                let _ = tls.shutdown().await;
            }
        });
        (url, bodies)
    }

    /// A test CA, a server certificate it signed for `127.0.0.1`, and a client certificate and key
    /// it signed, written to files.
    struct Pki {
        directory: tempfile::TempDir,
        server: rustls::ServerConfig,
        client_der: Vec<u8>,
    }

    fn pki(require_client: bool) -> Pki {
        use rcgen::{BasicConstraints, CertificateParams, CertifiedIssuer, IsCa, KeyPair};
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let mut ca_params = CertificateParams::new(Vec::<String>::new()).unwrap();
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let ca = CertifiedIssuer::self_signed(ca_params, KeyPair::generate().unwrap()).unwrap();
        let server_key = KeyPair::generate().unwrap();
        let server = CertificateParams::new(vec!["127.0.0.1".to_owned()])
            .unwrap()
            .signed_by(&server_key, &ca)
            .unwrap();
        let client_key = KeyPair::generate().unwrap();
        let client = CertificateParams::new(vec!["workload".to_owned()])
            .unwrap()
            .signed_by(&client_key, &ca)
            .unwrap();
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("ca.pem"), ca.pem()).unwrap();
        std::fs::write(directory.path().join("client.pem"), client.pem()).unwrap();
        std::fs::write(
            directory.path().join("client.key"),
            client_key.serialize_pem(),
        )
        .unwrap();
        let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
        let builder = rustls::ServerConfig::builder_with_provider(provider.clone())
            .with_safe_default_protocol_versions()
            .unwrap();
        let mut roots = rustls::RootCertStore::empty();
        roots.add(ca.der().clone()).unwrap();
        let builder = match require_client {
            true => builder.with_client_cert_verifier(
                rustls::server::WebPkiClientVerifier::builder_with_provider(
                    Arc::new(roots),
                    provider,
                )
                .build()
                .unwrap(),
            ),
            false => builder.with_no_client_auth(),
        };
        let server = builder
            .with_single_cert(
                vec![server.der().clone()],
                rustls::pki_types::PrivateKeyDer::try_from(server_key.serialize_der()).unwrap(),
            )
            .unwrap();
        Pki {
            directory,
            server,
            client_der: client.der().to_vec(),
        }
    }

    #[rstest]
    #[tokio::test]
    async fn a_certificate_source_exchanges_its_certificate_over_mutual_tls() {
        let pki = pki(true);
        let (url, bodies) = tls_token_endpoint(pki.server.clone()).await;
        let directory = pki.directory.path();
        let config = directory.join("certificate_config.json");
        std::fs::write(
            &config,
            json!({"cert_configs": {"workload": {
                "cert_path": directory.join("client.pem"), "key_path": directory.join("client.key"),
            }}})
            .to_string(),
        )
        .unwrap();
        let credentials = json!({
            "type": "external_account", "audience": POOL_AUDIENCE,
            "subject_token_type": "urn:ietf:params:oauth:token-type:mtls", "token_url": url,
            "credential_source": {"certificate": {"certificate_config_location": config}},
        });
        let login = auth(environment(&[(
            "REQUESTS_CA_BUNDLE",
            directory.join("ca.pem").to_str().unwrap(),
        )]))
        .login(Some(credentials.to_string()), Some("p".into()))
        .await
        .unwrap();
        assert_eq!(login.token.expose(), "granted");
        let body = bodies.lock().unwrap()[0].clone();
        let subject: Vec<String> = serde_json::from_str(
            &url::form_urlencoded::parse(body.as_bytes())
                .find(|(name, _)| name == "subject_token")
                .unwrap()
                .1,
        )
        .unwrap();
        assert_eq!(subject, [STANDARD.encode(&pki.client_der)]);
    }

    #[rstest]
    #[case::requests_bundle_only(&[("REQUESTS_CA_BUNDLE", "/etc/corporate-ca.pem")], true)]
    #[case::curl_bundle_only(&[("CURL_CA_BUNDLE", "/etc/corporate-ca.pem")], true)]
    #[case::ssl_cert_file_set(&[("REQUESTS_CA_BUNDLE", "/etc/corporate-ca.pem"), ("SSL_CERT_FILE", "/etc/corporate-ca.pem")], false)]
    #[case::no_bundle(&[], false)]
    #[tokio::test]
    async fn an_untrusted_token_endpoint_says_to_set_ssl_cert_file(
        #[case] env: &[(&'static str, &str)],
        #[case] hinted: bool,
    ) {
        let (url, _) = tls_token_endpoint(pki(false).server).await;
        let error = auth(FakeEnvironment {
            token_endpoint: url,
            ..environment(env)
        })
        .login(
            Some(authorized_user(json!({})).to_string()),
            Some("p".into()),
        )
        .await
        .unwrap_err();
        let failure = failure(error);
        assert_eq!(failure.exception, PythonException::Transport);
        assert_eq!(
            failure
                .message
                .contains("set SSL_CERT_FILE to the same CA bundle"),
            hinted,
            "{}",
            failure.message
        );
    }

    #[test]
    fn a_token_failure_never_quotes_the_sdk_error() {
        let error = std::io::Error::other(r#"missing field in {"saml": "secret-assertion"}"#);

        let failure = GoogleLoginFailure::token(&error, &FakeEnvironment::default());

        assert_eq!(failure.exception, PythonException::Refresh);
        assert!(!failure.message.contains("secret-assertion"));
    }

    const LOAD_FAILED: &str = "Failed to load vertex credentials. Check to see if credentials containing partial/invalid information. Error: ";

    #[rstest]
    #[case::credentials_that_do_not_load(
        json!({"type": "gdch_service_account", "client_email": CLIENT_EMAIL}),
        None,
        Some("project"),
        Some("The service account key is missing token_uri, private_key.")
    )]
    #[case::a_first_token_that_is_refused(
        authorized_user(json!({})),
        refused(400, json!({"error": "invalid_grant", "error_description": "secret-refresh-token-echo"})),
        Some("project"),
        Some("Vertex: Google refused or could not complete the token refresh.")
    )]
    #[case::a_project_that_does_not_resolve(authorized_user(json!({})), None, None, Some("Could not resolve project_id"))]
    #[case::a_login_that_succeeds(authorized_user(json!({})), None, Some("project"), None)]
    #[tokio::test]
    async fn a_failed_load_auth_is_logged_as_python_logs_it(
        #[future] google: Google,
        #[case] credentials: Value,
        #[case] answer: Option<(u16, Value)>,
        #[case] project: Option<&str>,
        #[case] logged: Option<&str>,
    ) {
        let google = google.await;
        if let Some((status, body)) = answer {
            google.queue(status, body);
        }
        let errors = Errors::default();

        let login = litellm_tracing::Logger::new(errors.clone())
            .instrument(
                auth(at(&google)).login(Some(credentials.to_string()), project.map(str::to_owned)),
            )
            .await;

        assert_eq!(login.is_ok(), logged.is_none());
        let records = errors.0.lock().unwrap().clone();
        assert_eq!(
            records,
            logged
                .map(|error| format!("{LOAD_FAILED}{error}"))
                .into_iter()
                .collect::<Vec<_>>()
        );
        assert!(records.iter().all(|record| !record.contains("secret")));
    }

    #[rstest]
    #[tokio::test]
    async fn credentials_served_from_the_cache_log_only_their_own_failures(
        #[future] google: Google,
    ) {
        let google = google.await;
        let auth = auth(at(&google));
        let credentials = authorized_user(json!({})).to_string();
        auth.login(Some(credentials.clone()), Some("project".into()))
            .await
            .unwrap();
        let errors = Errors::default();

        let login = litellm_tracing::Logger::new(errors.clone())
            .instrument(auth.login(Some(credentials), None))
            .await;

        assert_eq!(
            failure(login.unwrap_err()).message,
            "Could not resolve project_id"
        );
        assert_eq!(
            *errors.0.lock().unwrap(),
            [format!("{LOAD_FAILED}Could not resolve project_id")]
        );
    }
}
