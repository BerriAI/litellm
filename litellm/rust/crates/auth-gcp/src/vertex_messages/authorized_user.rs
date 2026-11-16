//! google-auth's `oauth2.credentials.Credentials` from authorized user info: the refresh token
//! exchanged at Google's token endpoint through `reauth.refresh_grant`, a form post like every
//! other google-auth token request.

use std::{
    sync::{Arc, Mutex, PoisonError},
    time::Duration,
};

use serde_json::{Map, Value};
use time::OffsetDateTime;

use super::{
    credentials::{Credentials, Loaded, Minted, RefreshFuture},
    failure::{GoogleLoginFailure, PythonException, python_str, python_type_name},
    service_account::{expiry, granted, refused, response_data, token_endpoint_request},
    transport::Requests,
};
use crate::constants::CLOUD_PLATFORM_SCOPE;

const REQUIRED_FIELDS: [&str; 3] = ["refresh_token", "client_id", "client_secret"];
const DEFAULT_UNIVERSE_DOMAIN: &str = "googleapis.com";
/// `_exponential_backoff`'s first wait.
const FIRST_RETRY_WAIT: Duration = Duration::from_secs(1);
const CANNOT_REFRESH: &str = "The credentials do not contain the necessary fields need to refresh the access token. You must specify refresh_token, token_uri, client_id, and client_secret.";
const REAUTH_NEEDED: &str = "Reauthentication is needed. Please run `gcloud auth application-default login` to reauthenticate.";

pub(crate) struct AuthorizedUser {
    refresh_token: Option<String>,
    client_id: Option<String>,
    client_secret: Option<String>,
    rapt_token: Option<String>,
    universe_domain: Option<String>,
    token_uri: String,
    /// The refresh token the last grant rotated to.
    rotated: Mutex<Option<String>>,
    first_retry_wait: Duration,
    requests: Requests,
}

/// `Credentials.from_authorized_user_info(info, scopes=[cloud-platform])`, which always refreshes
/// at `token_endpoint` whatever `token_uri` the info names.
pub(crate) fn authorized_user(
    info: &Map<String, Value>,
    token_endpoint: &str,
) -> Result<Loaded, GoogleLoginFailure> {
    Ok(Arc::new(load(
        info,
        token_endpoint,
        FIRST_RETRY_WAIT,
        Requests::process(),
    )?))
}

/// A field Python keeps as given: `None` for null, its `str()` otherwise.
fn field(info: &Map<String, Value>, name: &str) -> Option<String> {
    info.get(name)
        .filter(|value| !value.is_null())
        .map(|value| python_str(Some(value)))
}

/// `datetime.strptime(expiry.rstrip("Z").split(".")[0], "%Y-%m-%dT%H:%M:%S")` for a truthy expiry.
fn check_expiry(info: &Map<String, Value>) -> Result<(), GoogleLoginFailure> {
    let text = match info.get("expiry") {
        None | Some(Value::Null) | Some(Value::Bool(false)) => return Ok(()),
        Some(Value::String(text)) if text.is_empty() => return Ok(()),
        Some(Value::Number(number)) if number.as_f64() == Some(0.0) => return Ok(()),
        Some(Value::String(text)) => text,
        Some(other) => {
            return Err(GoogleLoginFailure::new(
                PythonException::Attribute,
                format!(
                    "'{}' object has no attribute 'rstrip'",
                    python_type_name(Some(other))
                ),
            ));
        }
    };
    let seconds = text
        .trim_end_matches('Z')
        .split('.')
        .next()
        .unwrap_or_default();
    time::PrimitiveDateTime::parse(
        seconds,
        time::macros::format_description!("[year]-[month]-[day]T[hour]:[minute]:[second]"),
    )
    .map(|_| ())
    .map_err(|_| {
        GoogleLoginFailure::new(
            PythonException::Value,
            format!("time data '{seconds}' does not match format '%Y-%m-%dT%H:%M:%S'"),
        )
    })
}

fn load(
    info: &Map<String, Value>,
    token_endpoint: &str,
    first_retry_wait: Duration,
    requests: Requests,
) -> Result<AuthorizedUser, GoogleLoginFailure> {
    let missing: Vec<&str> = REQUIRED_FIELDS
        .iter()
        .copied()
        .filter(|name| !info.contains_key(*name))
        .collect();
    if !missing.is_empty() {
        return Err(GoogleLoginFailure::new(
            PythonException::Value,
            format!(
                "Authorized user info was not in the expected format, missing fields {}.",
                missing.join(", ")
            ),
        ));
    }
    check_expiry(info)?;
    Ok(AuthorizedUser {
        refresh_token: field(info, "refresh_token"),
        client_id: field(info, "client_id"),
        client_secret: field(info, "client_secret"),
        rapt_token: field(info, "rapt_token").filter(|token| !token.is_empty()),
        universe_domain: match info.get("universe_domain") {
            None => Some(DEFAULT_UNIVERSE_DOMAIN.to_owned()),
            Some(_) => field(info, "universe_domain"),
        },
        token_uri: token_endpoint.to_owned(),
        rotated: Mutex::new(None),
        first_retry_wait,
        requests,
    })
}

impl AuthorizedUser {
    fn rotated(&self) -> std::sync::MutexGuard<'_, Option<String>> {
        self.rotated.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// `Credentials._perform_refresh_token` through `reauth.refresh_grant`.
    async fn mint(&self) -> Result<Minted, GoogleLoginFailure> {
        if self.universe_domain.as_deref() != Some(DEFAULT_UNIVERSE_DOMAIN) {
            return Err(GoogleLoginFailure::new(
                PythonException::Refresh,
                format!(
                    "User credential refresh is only supported in the default googleapis.com universe domain, but the current universe domain is {}. If you created the credential with an access token, it's likely that the provided token is expired now, please update your code with a valid token.",
                    self.universe_domain.as_deref().unwrap_or("None")
                ),
            ));
        }
        let refresh_token = self
            .rotated()
            .clone()
            .or_else(|| self.refresh_token.clone());
        let (Some(refresh_token), Some(client_id), Some(client_secret)) =
            (refresh_token, &self.client_id, &self.client_secret)
        else {
            return Err(GoogleLoginFailure::new(
                PythonException::Refresh,
                CANNOT_REFRESH,
            ));
        };
        let mut form = vec![
            ("grant_type", "refresh_token"),
            ("client_id", client_id.as_str()),
            ("client_secret", client_secret.as_str()),
            ("refresh_token", refresh_token.as_str()),
            ("scope", CLOUD_PLATFORM_SCOPE),
        ];
        if let Some(rapt) = &self.rapt_token {
            form.push(("rapt", rapt.as_str()));
        }
        let (status, body) = token_endpoint_request(
            &self.requests,
            &self.token_uri,
            &form,
            self.first_retry_wait,
        )
        .await?;
        if status != 200 {
            return Err(refusal(&body));
        }
        let granted = granted(&body)?;
        if let Some(rotated) = granted.get("refresh_token") {
            *self.rotated() = Some(python_str(Some(rotated)));
        }
        Ok(Minted::from_reply(
            granted.get("access_token"),
            expiry(&granted, OffsetDateTime::now_utc())?,
        ))
    }
}

/// `reauth.refresh_grant` on a refused answer: a text body is the error itself, an
/// `invalid_grant` with a rapt subtype asks to log in again, anything but a dict crashes on
/// `.get`, and the rest is `_handle_error_response`'s.
fn refusal(body: &str) -> GoogleLoginFailure {
    let data = response_data(body);
    let fields = match &data {
        Value::String(text) => return GoogleLoginFailure::new(PythonException::Refresh, text),
        Value::Object(fields) => fields,
        other => {
            return GoogleLoginFailure::new(
                PythonException::Attribute,
                format!(
                    "'{}' object has no attribute 'get'",
                    python_type_name(Some(other))
                ),
            );
        }
    };
    let subtype = fields.get("error_subtype").and_then(Value::as_str);
    if fields.get("error").and_then(Value::as_str) == Some("invalid_grant")
        && matches!(subtype, Some("invalid_rapt" | "rapt_required"))
    {
        return GoogleLoginFailure::new(PythonException::Refresh, REAUTH_NEEDED);
    }
    refused(body)
}

impl Credentials for AuthorizedUser {
    fn refresh(&self) -> RefreshFuture<'_> {
        Box::pin(self.mint())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::vertex_messages::service_account::tests::endpoint;

    fn info(extra: Value) -> Map<String, Value> {
        json!({
            "type": "authorized_user", "client_id": "fz-client", "client_secret": "fz-secret",
            "refresh_token": "fz-refresh",
        })
        .as_object()
        .unwrap()
        .clone()
        .into_iter()
        .chain(extra.as_object().unwrap().clone())
        .collect()
    }

    fn user(url: &str, extra: Value) -> AuthorizedUser {
        load(
            &info(extra),
            url,
            Duration::from_millis(1),
            Requests::with_environment(&[]),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn the_refresh_is_google_auth_s_form_post() {
        let endpoint = endpoint(vec![(
            200,
            json!({"access_token": "au-token", "expires_in": 3599, "refresh_token": "rotated"}),
        )])
        .await;
        let user = user(&endpoint.url, json!({"rapt_token": "rapt-1"}));

        assert_eq!(user.mint().await.unwrap().value.expose(), "au-token");
        user.mint().await.unwrap();

        let scope = "https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fcloud-platform";
        assert_eq!(
            endpoint.forms.lock().unwrap().clone(),
            [
                format!(
                    "grant_type=refresh_token&client_id=fz-client&client_secret=fz-secret&refresh_token=fz-refresh&scope={scope}&rapt=rapt-1"
                ),
                format!(
                    "grant_type=refresh_token&client_id=fz-client&client_secret=fz-secret&refresh_token=rotated&scope={scope}&rapt=rapt-1"
                ),
            ]
        );
        assert_eq!(
            endpoint.content_types.lock().unwrap()[0].as_deref(),
            Some("application/x-www-form-urlencoded")
        );
        assert_eq!(endpoint.accepts.lock().unwrap()[0].as_deref(), Some("*/*"));
    }

    #[rstest]
    #[case::invalid_grant(
        json!({"error": "invalid_grant", "error_description": "Token has been expired or revoked."}),
        PythonException::Refresh,
        "('invalid_grant: Token has been expired or revoked.', {'error': 'invalid_grant', 'error_description': 'Token has been expired or revoked.'})"
    )]
    #[case::rapt(
        json!({"error": "invalid_grant", "error_subtype": "invalid_rapt"}),
        PythonException::Refresh,
        REAUTH_NEEDED
    )]
    #[case::text(json!("denied"), PythonException::Refresh, "denied")]
    #[case::list(json!(["denied"]), PythonException::Attribute, "'list' object has no attribute 'get'")]
    #[tokio::test]
    async fn a_refused_refresh_raises_what_refresh_grant_raises(
        #[case] answer: Value,
        #[case] exception: PythonException,
        #[case] message: &str,
    ) {
        let endpoint = endpoint(vec![(400, answer)]).await;
        assert_eq!(
            user(&endpoint.url, json!({})).mint().await.unwrap_err(),
            GoogleLoginFailure::new(exception, message)
        );
        assert_eq!(endpoint.forms.lock().unwrap().len(), 1);
    }

    #[rstest]
    #[case::null_refresh_token(json!({"refresh_token": null}), CANNOT_REFRESH)]
    #[case::other_universe(
        json!({"universe_domain": "example.com"}),
        "User credential refresh is only supported in the default googleapis.com universe domain, but the current universe domain is example.com. If you created the credential with an access token, it's likely that the provided token is expired now, please update your code with a valid token."
    )]
    #[tokio::test]
    async fn credentials_that_cannot_refresh_are_refused_without_a_request(
        #[case] extra: Value,
        #[case] message: &str,
    ) {
        let endpoint = endpoint(vec![(200, json!({"access_token": "t"}))]).await;
        assert_eq!(
            user(&endpoint.url, extra).mint().await.unwrap_err(),
            GoogleLoginFailure::new(PythonException::Refresh, message)
        );
        assert!(endpoint.forms.lock().unwrap().is_empty());
    }

    #[rstest]
    #[case::text(json!("2026-11-16T12:00:00.5Z"), None)]
    #[case::bad_text(
        json!("soon"),
        Some((PythonException::Value, "time data 'soon' does not match format '%Y-%m-%dT%H:%M:%S'"))
    )]
    #[case::number(json!(5), Some((PythonException::Attribute, "'int' object has no attribute 'rstrip'")))]
    fn a_saved_expiry_is_parsed_at_load(
        #[case] expiry: Value,
        #[case] expected: Option<(PythonException, &str)>,
    ) {
        let loaded = load(
            &info(json!({ "expiry": expiry })),
            "http://127.0.0.1:9/token",
            Duration::from_millis(1),
            Requests::with_environment(&[]),
        );
        assert_eq!(
            loaded.err(),
            expected.map(|(exception, message)| GoogleLoginFailure::new(exception, message))
        );
    }
}
