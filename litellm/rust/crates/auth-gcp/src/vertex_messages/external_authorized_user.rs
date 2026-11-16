//! google-auth's `external_account_authorized_user.Credentials`, a workforce identity user's
//! refresh token exchanged at the STS endpoint its file names.

use std::{
    sync::{Arc, Mutex, PoisonError},
    time::Duration,
};

use base64::{Engine, engine::general_purpose::STANDARD};
use serde_json::{Map, Value};
use time::{OffsetDateTime, PrimitiveDateTime, macros::format_description};

use super::{
    builders::token_failure,
    credentials::{Credentials, Expiry, Loaded, Minted, REFRESH_THRESHOLD, RefreshFuture},
    failure::{GoogleLoginFailure, PythonException, python_repr, python_str},
    transport::Requests,
};

const CANNOT_REFRESH: &str = "The credentials do not contain the necessary fields need to refresh the access token. You must specify refresh_token, token_url, client_id, and client_secret.";

/// The token a file holds, known only by when it stops being valid.
struct Token {
    expiry: Option<OffsetDateTime>,
}

impl Token {
    /// `credentials.valid`: a token that is not within the refresh threshold of its expiry.
    fn valid(&self) -> bool {
        self.expiry
            .is_none_or(|expiry| OffsetDateTime::now_utc() < expiry - REFRESH_THRESHOLD)
    }
}

struct Refresh {
    refresh_token: String,
    token_url: String,
    client_id: String,
    client_secret: String,
}

/// Python's Vertex login refreshes the credentials it loads before it takes their token, so the
/// token a file holds only decides whether the credentials load.
struct ExternalAuthorizedUser {
    refresh: Option<Refresh>,
    /// The refresh token the last exchange rotated to.
    rotated: Mutex<Option<String>>,
    requests: Requests,
}

fn truthy(info: &Map<String, Value>, field: &str) -> Option<String> {
    match info.get(field) {
        None | Some(Value::Null) | Some(Value::Bool(false)) => None,
        Some(Value::String(text)) if text.is_empty() => None,
        value => Some(python_str(value)),
    }
}

/// `datetime.strptime(expiry.rstrip("Z").split(".")[0], "%Y-%m-%dT%H:%M:%S")`, a naive UTC time.
fn expiry(text: &str) -> Option<OffsetDateTime> {
    let seconds = text.trim_end_matches('Z').split('.').next()?;
    PrimitiveDateTime::parse(
        seconds,
        format_description!("[year]-[month]-[day]T[hour]:[minute]:[second]"),
    )
    .ok()
    .map(PrimitiveDateTime::assume_utc)
}

/// `Credentials.from_info`, refused as `_get_external_account_authorized_user_credentials`
/// refuses any `ValueError` it raises.
pub(crate) fn external_account_authorized_user(
    info: &Map<String, Value>,
    file: &str,
) -> Result<Loaded, GoogleLoginFailure> {
    let unloadable = || {
        GoogleLoginFailure::new(
            PythonException::DefaultCredentials,
            format!("Failed to load external account authorized user credentials from {file}"),
        )
    };
    let expiry = match info.get("expiry") {
        Some(Value::String(text)) if !text.is_empty() => Some(expiry(text).ok_or_else(unloadable)?),
        _ => None,
    };
    let file_token = info
        .get("token")
        .filter(|token| !token.is_null())
        .map(|_| Token { expiry });
    let refresh = match (
        truthy(info, "refresh_token"),
        truthy(info, "token_url"),
        truthy(info, "client_id"),
        truthy(info, "client_secret"),
    ) {
        (Some(refresh_token), Some(token_url), Some(client_id), Some(client_secret)) => {
            Some(Refresh {
                refresh_token,
                token_url,
                client_id,
                client_secret,
            })
        }
        _ => None,
    };
    if refresh.is_none() && !file_token.as_ref().is_some_and(Token::valid) {
        return Err(unloadable());
    }
    Ok(Arc::new(ExternalAuthorizedUser {
        rotated: Mutex::new(None),
        refresh,
        requests: Requests::process(),
    }))
}

/// `utils.handle_error_response`'s `OAuthError(details, body)`.
pub(crate) fn oauth_error(body: &str) -> GoogleLoginFailure {
    let details = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|error| {
            let code = error.get("error")?;
            Some(format!(
                "Error code {}{}{}",
                python_str(Some(code)),
                error
                    .get("error_description")
                    .map_or_else(String::new, |text| format!(": {}", python_str(Some(text)))),
                error
                    .get("error_uri")
                    .map_or_else(String::new, |text| format!(" - {}", python_str(Some(text)))),
            ))
        })
        .unwrap_or_else(|| body.to_owned());
    GoogleLoginFailure::new(
        PythonException::OAuth,
        format!("({}, {})", python_repr(&details), python_repr(body)),
    )
}

impl ExternalAuthorizedUser {
    /// `_perform_refresh_token` through `sts.Client.refresh_token`.
    async fn exchange(&self, refresh: &Refresh) -> Result<Minted, GoogleLoginFailure> {
        let refresh_token = self
            .lock()
            .clone()
            .unwrap_or_else(|| refresh.refresh_token.clone());
        let client = self.requests.client(&refresh.token_url)?;
        let credentials =
            STANDARD.encode(format!("{}:{}", refresh.client_id, refresh.client_secret));
        let response = client
            .post(&refresh.token_url)
            .header("Authorization", format!("Basic {credentials}"))
            .form(&[
                ("grant_type", "refresh_token"),
                ("refresh_token", refresh_token.as_str()),
            ])
            .send()
            .await
            .map_err(|error| token_failure(&error))?;
        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|error| token_failure(&error))?;
        if status != reqwest::StatusCode::OK {
            return Err(oauth_error(&body));
        }
        let granted: Map<String, Value> = serde_json::from_str(&body)
            .map_err(|error| GoogleLoginFailure::new(PythonException::Value, error.to_string()))?;
        let lifetime = granted
            .get("expires_in")
            .and_then(Value::as_f64)
            .ok_or_else(|| {
                GoogleLoginFailure::new(
                    PythonException::Type,
                    "unsupported type for timedelta seconds component: NoneType",
                )
            })?;
        if let Some(rotated) = granted.get("refresh_token") {
            *self.lock() = Some(python_str(Some(rotated)));
        }
        Ok(Minted::from_reply(
            granted.get("access_token"),
            Expiry::At(OffsetDateTime::now_utc() + Duration::from_secs_f64(lifetime.max(0.0))),
        ))
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<String>> {
        self.rotated.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Credentials for ExternalAuthorizedUser {
    fn refresh(&self) -> RefreshFuture<'_> {
        Box::pin(async {
            let Some(refresh) = &self.refresh else {
                return Err(GoogleLoginFailure::new(
                    PythonException::Refresh,
                    CANNOT_REFRESH,
                ));
            };
            self.exchange(refresh).await
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::{Router, http::HeaderMap, routing::post};
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    type Seen = Arc<Mutex<Vec<(Option<String>, String)>>>;

    /// An STS endpoint that answers each request with the next scripted status and body, and
    /// records the `Authorization` header and form body it was sent.
    async fn sts(answers: Vec<(u16, Value)>) -> (String, Seen, tokio::task::JoinHandle<()>) {
        let seen: Seen = Arc::default();
        let recorded = seen.clone();
        let answers = Arc::new(Mutex::new(answers.into_iter()));
        let app = Router::new().route(
            "/sts",
            post(move |headers: HeaderMap, body: String| {
                let recorded = recorded.clone();
                let answers = answers.clone();
                async move {
                    let authorization = headers
                        .get("authorization")
                        .map(|value| value.to_str().unwrap().to_owned());
                    recorded.lock().unwrap().push((authorization, body));
                    let (status, body) = answers.lock().unwrap().next().unwrap();
                    (
                        axum::http::StatusCode::from_u16(status).unwrap(),
                        body.to_string(),
                    )
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/sts", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (url, seen, server)
    }

    fn user(fields: Value) -> Map<String, Value> {
        json!({"type": "external_account_authorized_user", "audience": "//iam.googleapis.com/locations/global/workforcePools/p/providers/q"})
            .as_object()
            .unwrap()
            .clone()
            .into_iter()
            .chain(fields.as_object().unwrap().clone())
            .collect()
    }

    fn refreshable(url: &str) -> Map<String, Value> {
        user(json!({
            "refresh_token": "refresh-1", "token_url": url,
            "client_id": "client", "client_secret": "secret", "token": "file-token",
        }))
    }

    async fn failure(source: &Loaded) -> GoogleLoginFailure {
        source.refresh().await.unwrap_err()
    }

    #[tokio::test]
    async fn the_refresh_token_is_exchanged_with_basic_client_auth_and_rotated() {
        let (url, seen, _server) = sts(vec![
            (
                200,
                json!({"access_token": "sts-1", "expires_in": 0, "refresh_token": "refresh-2"}),
            ),
            (200, json!({"access_token": "sts-2", "expires_in": 3600})),
        ])
        .await;
        let source = external_account_authorized_user(&refreshable(&url), "/k.json").unwrap();

        let first = source.refresh().await.unwrap();
        let second = source.refresh().await.unwrap();

        assert_eq!(
            (first.value.expose(), second.value.expose()),
            ("sts-1", "sts-2")
        );
        let Expiry::At(expiry) = second.expiry else {
            panic!("a granted token expires");
        };
        assert!(expiry > OffsetDateTime::now_utc() + Duration::from_secs(3500));
        let basic = format!("Basic {}", STANDARD.encode("client:secret"));
        assert_eq!(
            *seen.lock().unwrap(),
            vec![
                (
                    Some(basic.clone()),
                    "grant_type=refresh_token&refresh_token=refresh-1".to_owned()
                ),
                (
                    Some(basic),
                    "grant_type=refresh_token&refresh_token=refresh-2".to_owned()
                ),
            ]
        );
    }

    #[rstest]
    #[case::described(
        json!({"error": "invalid_grant", "error_description": "expired", "error_uri": "https://e"}),
        "('Error code invalid_grant: expired - https://e', '{\"error\":\"invalid_grant\",\"error_description\":\"expired\",\"error_uri\":\"https://e\"}')"
    )]
    #[case::not_an_oauth_error(
        json!({"message": "no"}),
        "('{\"message\":\"no\"}', '{\"message\":\"no\"}')"
    )]
    #[tokio::test]
    async fn a_refused_refresh_raises_python_s_oauth_error(
        #[case] answer: Value,
        #[case] expected: &str,
    ) {
        let (url, _seen, _server) = sts(vec![(400, answer)]).await;
        let source = external_account_authorized_user(&refreshable(&url), "/k.json").unwrap();
        assert_eq!(
            failure(&source).await,
            GoogleLoginFailure::new(PythonException::OAuth, expected)
        );
    }

    #[tokio::test]
    async fn a_token_without_refresh_fields_loads_but_cannot_be_refreshed() {
        let source =
            external_account_authorized_user(&user(json!({"token": "file-token"})), "/k.json")
                .unwrap();
        assert_eq!(
            failure(&source).await,
            GoogleLoginFailure::new(PythonException::Refresh, CANNOT_REFRESH)
        );
    }

    #[rstest]
    #[case::nothing_to_use(json!({}))]
    #[case::expired_token(json!({"token": "t", "expiry": "2001-01-01T00:00:00Z"}))]
    #[case::unparsable_expiry(json!({"token": "t", "expiry": "tomorrow", "refresh_token": "r", "token_url": "u", "client_id": "c", "client_secret": "s"}))]
    #[case::refresh_field_missing(json!({"refresh_token": "r", "token_url": "u", "client_id": "c"}))]
    fn credentials_python_cannot_build_fail_to_load(#[case] fields: Value) {
        let Err(error) = external_account_authorized_user(&user(fields), "/k.json") else {
            panic!("the credentials should not load");
        };
        assert_eq!(
            error,
            GoogleLoginFailure::new(
                PythonException::DefaultCredentials,
                "Failed to load external account authorized user credentials from /k.json"
            )
        );
    }

    #[rstest]
    #[case::fractional_utc("2999-01-02T03:04:05.123Z")]
    #[case::plain("2999-01-02T03:04:05")]
    fn an_unexpired_token_loads_without_refresh_fields(#[case] expiry: &str) {
        assert!(
            external_account_authorized_user(
                &user(json!({"token": "t", "expiry": expiry})),
                "/k.json"
            )
            .is_ok()
        );
    }
}
