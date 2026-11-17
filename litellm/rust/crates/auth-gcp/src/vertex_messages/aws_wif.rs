//! LiteLLM's explicit AWS login and google-auth's metadata AWS login for an external account:
//! a GetCallerIdentity request signed with AWS credentials, serialized as the subject token STS
//! exchanges for a Google token.

use std::time::SystemTime;

use http::{HeaderMap, HeaderValue, header::AUTHORIZATION};
use litellm_auth_aws::{AwsAuthConfig, AwsAuthService, SigningScope, sign_headers};
use litellm_python_compat::urllib::quote;
use litellm_secret_value::SecretValue;
use serde::Deserialize;
use serde_json::json;

use super::{
    builders::token_failure,
    credentials::Loaded,
    external_account::{
        ExternalAccount, ExternalAccountInfo, OsEnvironment, Settings, SubjectFuture, SubjectToken,
    },
    failure::{GoogleLoginFailure, PythonException},
    source::present,
    transport::Requests,
};

/// LiteLLM's own `aws_*` keys in an external account file. Each is `Some(None)` when null, which
/// still chooses the explicit AWS login.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub(crate) struct ExplicitAwsKeys {
    #[serde(default, deserialize_with = "present")]
    aws_access_key_id: Option<Option<String>>,
    #[serde(default, deserialize_with = "present")]
    aws_secret_access_key: Option<Option<String>>,
    #[serde(default, deserialize_with = "present")]
    aws_session_token: Option<Option<String>>,
    #[serde(default, deserialize_with = "present")]
    aws_region_name: Option<Option<String>>,
    #[serde(default, deserialize_with = "present")]
    aws_session_name: Option<Option<String>>,
    #[serde(default, deserialize_with = "present")]
    aws_profile_name: Option<Option<String>>,
    #[serde(default, deserialize_with = "present")]
    aws_role_name: Option<Option<String>>,
    #[serde(default, deserialize_with = "present")]
    aws_web_identity_token: Option<Option<String>>,
    #[serde(default, deserialize_with = "present")]
    aws_sts_endpoint: Option<Option<String>>,
    #[serde(default, deserialize_with = "present")]
    aws_external_id: Option<Option<String>>,
}

impl ExplicitAwsKeys {
    fn any_given(&self) -> bool {
        [
            &self.aws_access_key_id,
            &self.aws_secret_access_key,
            &self.aws_session_token,
            &self.aws_region_name,
            &self.aws_session_name,
            &self.aws_profile_name,
            &self.aws_role_name,
            &self.aws_web_identity_token,
            &self.aws_sts_endpoint,
            &self.aws_external_id,
        ]
        .into_iter()
        .any(Option::is_some)
    }
}

/// A key's text when it is set and not empty.
fn text(field: &Option<Option<String>>) -> Option<String> {
    field.clone().flatten().filter(|value| !value.is_empty())
}
const MISSING_REGION: &str = "aws_region_name is required in the WIF credential JSON when using explicit AWS authentication. Add \"aws_region_name\": \"<your-region>\" to your credential file.";
/// `aws._DEFAULT_AWS_REGIONAL_CREDENTIAL_VERIFICATION_URL`.
const CALLER_IDENTITY_URL: &str =
    "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15";
const TARGET_RESOURCE_HEADER: &str = "x-goog-cloud-target-resource";
/// `aws._IMDSV2_SESSION_TOKEN_TTL_SECONDS`.
const IMDSV2_SESSION_TOKEN_TTL_SECONDS: &str = "300";

/// Python's `_credentials_from_json` routes an `external_account` whose `environment_id` names
/// AWS and that carries any of LiteLLM's `aws_*` keys, even a null one, to
/// `VertexAIAwsWifAuth.credentials_from_explicit_aws`.
pub(crate) fn uses_explicit_aws(info: &ExternalAccountInfo) -> bool {
    names_aws(info) && info.aws.any_given()
}

/// An external account whose `credential_source.environment_id` names AWS.
pub(crate) fn names_aws(info: &ExternalAccountInfo) -> bool {
    info.credential_source
        .as_ref()
        .and_then(|source| source.environment_id.as_deref())
        .is_some_and(|environment_id| environment_id.contains("aws"))
}

/// Python's `VertexAIAwsWifAuth.credentials_from_explicit_aws`: the subject token is a
/// GetCallerIdentity request signed with the AWS credentials LiteLLM's own `aws_*` keys
/// resolve, never the EC2 metadata service.
pub(crate) fn explicit_aws_credentials(
    info: &ExternalAccountInfo,
    os_env: OsEnvironment,
) -> Result<Loaded, GoogleLoginFailure> {
    let keys = &info.aws;
    let region = text(&keys.aws_region_name)
        .ok_or_else(|| GoogleLoginFailure::new(PythonException::Value, MISSING_REGION))?;
    let secret = |field: &Option<Option<String>>| text(field).map(SecretValue::new);
    let config = AwsAuthConfig {
        access_key_id: secret(&keys.aws_access_key_id),
        secret_access_key: secret(&keys.aws_secret_access_key),
        session_token: secret(&keys.aws_session_token),
        region_name: Some(region.clone()),
        session_name: text(&keys.aws_session_name),
        profile_name: text(&keys.aws_profile_name),
        role_name: text(&keys.aws_role_name),
        web_identity_token: secret(&keys.aws_web_identity_token),
        sts_endpoint: text(&keys.aws_sts_endpoint),
        external_id: secret(&keys.aws_external_id),
        ..AwsAuthConfig::default()
    };
    let audience = info
        .audience
        .clone()
        .filter(|audience| !audience.is_empty())
        .unwrap_or_default();
    let subject = SignedCallerIdentity {
        aws: AwsAuthService::default(),
        config,
        region,
        audience,
        os_env,
    };
    Ok(ExternalAccount::loaded(
        Settings::for_explicit_aws(info),
        Box::new(subject),
    ))
}

/// The subject token of LiteLLM's explicit AWS login: a GetCallerIdentity request signed with
/// the credentials LiteLLM's AWS login resolves from the file's `aws_*` keys.
struct SignedCallerIdentity {
    aws: AwsAuthService,
    config: AwsAuthConfig,
    region: String,
    audience: String,
    os_env: OsEnvironment,
}

impl SubjectToken for SignedCallerIdentity {
    fn subject_token<'a>(&'a self, _requests: &'a Requests) -> SubjectFuture<'a> {
        Box::pin(async move {
            let os_env = self.os_env.clone();
            let credentials = self
                .aws
                .resolve_credentials(self.config.clone(), &move |name| os_env(name))
                .await
                .map_err(|error| token_failure(&error))?;
            caller_identity_subject_token(
                &credentials,
                &self.region,
                &CALLER_IDENTITY_URL.replace("{region}", &self.region),
                &self.audience,
                SystemTime::now(),
            )
            .map_err(|error| token_failure(&error))
        })
    }
}

fn refresh_error(message: impl Into<String>) -> GoogleLoginFailure {
    GoogleLoginFailure::new(PythonException::Refresh, message)
}

/// google-auth's `_DefaultAwsSecurityCredentialsSupplier`: the region and keys from the
/// environment, else from the EC2 metadata service, asked through `requests`.
struct MetadataCallerIdentity {
    region_url: Option<String>,
    security_credentials_url: Option<String>,
    imdsv2_session_token_url: Option<String>,
    verification_url: String,
    audience: String,
    os_env: OsEnvironment,
    /// The region the first subject token found, which google-auth's request signer keeps.
    region: tokio::sync::OnceCell<String>,
}

impl MetadataCallerIdentity {
    async fn get(
        &self,
        requests: &Requests,
        url: &str,
        session_token: Option<&str>,
    ) -> Result<(u16, String), GoogleLoginFailure> {
        let mut request = requests
            .client(url)?
            .get(url)
            .header(http::header::ACCEPT, "*/*");
        if let Some(token) = session_token {
            request = request.header("X-aws-ec2-metadata-token", token);
        }
        let response = request
            .send()
            .await
            .map_err(|error| token_failure(&error))?;
        let status = response.status().as_u16();
        let body = response
            .text()
            .await
            .map_err(|error| token_failure(&error))?;
        Ok((status, body))
    }

    /// `_get_imdsv2_session_token`.
    async fn session_token(
        &self,
        requests: &Requests,
    ) -> Result<Option<String>, GoogleLoginFailure> {
        let Some(url) = &self.imdsv2_session_token_url else {
            return Ok(None);
        };
        let response = requests
            .client(url)?
            .put(url)
            .header(http::header::ACCEPT, "*/*")
            .header(
                "X-aws-ec2-metadata-token-ttl-seconds",
                IMDSV2_SESSION_TOKEN_TTL_SECONDS,
            )
            .send()
            .await
            .map_err(|error| token_failure(&error))?;
        let status = response.status().as_u16();
        let body = response
            .text()
            .await
            .map_err(|error| token_failure(&error))?;
        if status != 200 {
            return Err(refresh_error(format!(
                "Unable to retrieve AWS Session Token: {}",
                body
            )));
        }
        Ok(Some(body))
    }

    /// `get_aws_region`.
    async fn find_region(&self, requests: &Requests) -> Result<String, GoogleLoginFailure> {
        if let Some(region) =
            (self.os_env)("AWS_REGION").or_else(|| (self.os_env)("AWS_DEFAULT_REGION"))
        {
            return Ok(region);
        }
        let Some(url) = self.region_url.as_deref().filter(|url| !url.is_empty()) else {
            return Err(refresh_error("Unable to determine AWS region"));
        };
        let session_token = self.session_token(requests).await?;
        let (status, body) = self.get(requests, url, session_token.as_deref()).await?;
        if status != 200 {
            return Err(refresh_error(format!(
                "Unable to retrieve AWS region: {body}"
            )));
        }
        let mut region = body;
        region.pop();
        Ok(region)
    }

    /// `get_aws_security_credentials`.
    async fn credentials(
        &self,
        requests: &Requests,
    ) -> Result<litellm_auth_aws::Credentials, GoogleLoginFailure> {
        let env = |name: &str| (self.os_env)(name).filter(|value| !value.is_empty());
        if let (Some(key), Some(secret)) = (env("AWS_ACCESS_KEY_ID"), env("AWS_SECRET_ACCESS_KEY"))
        {
            return Ok(litellm_auth_aws::Credentials::new(
                key,
                secret,
                (self.os_env)("AWS_SESSION_TOKEN"),
                None,
                "google-auth",
            ));
        }
        let session_token = self.session_token(requests).await?;
        let Some(url) = &self.security_credentials_url else {
            return Err(refresh_error(
                "Unable to determine the AWS metadata server security credentials endpoint",
            ));
        };
        let (status, role) = self.get(requests, url, session_token.as_deref()).await?;
        if status != 200 {
            return Err(refresh_error(format!(
                "Unable to retrieve AWS role name {role}"
            )));
        }
        let (status, body) = self
            .get(requests, &format!("{url}/{role}"), session_token.as_deref())
            .await?;
        if status != 200 {
            return Err(refresh_error(format!(
                "Unable to retrieve AWS security credentials: {body}"
            )));
        }
        let Ok(granted) = serde_json::from_str::<MetadataCredentials>(&body) else {
            return Err(refresh_error(
                "Unable to retrieve AWS security credentials: the metadata service's reply is not a JSON object",
            ));
        };
        Ok(litellm_auth_aws::Credentials::new(
            granted.access_key_id.unwrap_or_default(),
            granted.secret_access_key.unwrap_or_default(),
            granted.token,
            None,
            "google-auth",
        ))
    }
}

/// The keys the EC2 metadata service answers a role's security credentials with.
#[derive(Deserialize)]
struct MetadataCredentials {
    #[serde(rename = "AccessKeyId")]
    access_key_id: Option<String>,
    #[serde(rename = "SecretAccessKey")]
    secret_access_key: Option<String>,
    #[serde(rename = "Token")]
    token: Option<String>,
}

impl SubjectToken for MetadataCallerIdentity {
    fn subject_token<'a>(&'a self, requests: &'a Requests) -> SubjectFuture<'a> {
        Box::pin(async move {
            let region = self
                .region
                .get_or_try_init(|| self.find_region(requests))
                .await?
                .clone();
            let credentials = self.credentials(requests).await?;
            caller_identity_subject_token(
                &credentials,
                &region,
                &self.verification_url.replace("{region}", &region),
                &self.audience,
                SystemTime::now(),
            )
            .map_err(|error| token_failure(&error))
        })
    }
}

/// google-auth's `aws.Credentials.from_info` for a metadata-sourced AWS account.
pub(crate) fn metadata_aws_credentials(
    info: &ExternalAccountInfo,
    os_env: OsEnvironment,
) -> Result<Loaded, GoogleLoginFailure> {
    let settings = Settings::from_info(info)?;
    let Some(source) = &info.credential_source else {
        return Err(GoogleLoginFailure::new(
            PythonException::InvalidValue,
            "A valid credential source or AWS security credentials supplier must be provided.",
        ));
    };
    let environment_id = source.environment_id.clone().unwrap_or_default();
    let version = environment_id
        .strip_prefix("aws")
        .filter(|version| !version.is_empty() && version.bytes().all(|byte| byte.is_ascii_digit()));
    let verification_url = source.regional_cred_verification_url.clone();
    let (Some(version), Some(verification_url)) = (version, verification_url) else {
        return Err(GoogleLoginFailure::new(
            PythonException::InvalidResource,
            "No valid AWS 'credential_source' provided",
        ));
    };
    if version.parse::<u64>().ok() != Some(1) {
        return Err(GoogleLoginFailure::new(
            PythonException::InvalidValue,
            format!("aws version '{version}' is not supported in the current build."),
        ));
    }
    let subject = MetadataCallerIdentity {
        region_url: source.region_url.clone(),
        security_credentials_url: source.url.clone(),
        imdsv2_session_token_url: source.imdsv2_session_token_url.clone(),
        verification_url,
        audience: settings.audience.clone().unwrap_or_else(|| "None".into()),
        os_env,
        region: tokio::sync::OnceCell::new(),
    };
    Ok(ExternalAccount::loaded(settings, Box::new(subject)))
}

/// google-auth's `aws.Credentials.retrieve_subject_token`: the signed request's headers in the
/// order google-auth lists them, with the unsigned target resource last, serialized with sorted
/// keys and URL-quoted.
fn caller_identity_subject_token(
    credentials: &litellm_auth_aws::Credentials,
    region: &str,
    url: &str,
    audience: &str,
    time: SystemTime,
) -> Result<String, litellm_auth_aws::Error> {
    let host = url::Url::parse(url)
        .ok()
        .and_then(|url| url.host_str().map(str::to_owned))
        .unwrap_or_default();
    let unsigned = HeaderMap::from_iter([(
        http::header::HOST,
        HeaderValue::from_str(&host).map_err(|_| litellm_auth_aws::Error::NonAsciiHeader {
            name: "host".into(),
        })?,
    )]);
    let signed = sign_headers(
        "POST",
        url,
        &[],
        &unsigned,
        SigningScope {
            region,
            service: "sts",
            credentials,
            time,
        },
    )?;
    let header = |name: &str| {
        signed
            .get(name)
            .and_then(|value| value.to_str().ok())
            .map(|value| json!({"key": if name == AUTHORIZATION { "Authorization" } else { name }, "value": value}))
    };
    let headers: Vec<serde_json::Value> = [
        header(AUTHORIZATION.as_str()),
        Some(json!({"key": "host", "value": host})),
        header("x-amz-date"),
        header("x-amz-security-token"),
        Some(json!({"key": TARGET_RESOURCE_HEADER, "value": audience})),
    ]
    .into_iter()
    .flatten()
    .collect();
    let request = json!({"headers": headers, "method": "POST", "url": url}).to_string();
    Ok(quote(&request, "/"))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use serde_json::Value;
    use std::time::{Duration, UNIX_EPOCH};

    use super::*;

    fn aws_info(extra: Value) -> ExternalAccountInfo {
        let base = json!({
            "type": "external_account",
            "audience": "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/pool/providers/aws",
            "subject_token_type": "urn:ietf:params:aws:token-type:aws4_request",
            "credential_source": {"environment_id": "aws1"}
        });
        let info: serde_json::Map<String, Value> = base
            .as_object()
            .unwrap()
            .clone()
            .into_iter()
            .chain(extra.as_object().unwrap().clone())
            .collect();
        serde_json::from_value(Value::Object(info)).unwrap()
    }

    #[rstest]
    #[case::static_keys(aws_info(json!({"aws_access_key_id": "key"})), true)]
    #[case::null_key(aws_info(json!({"aws_role_name": null})), true)]
    #[case::metadata_only(aws_info(json!({})), false)]
    #[case::not_aws(
        serde_json::from_value(json!({"aws_access_key_id": "key", "credential_source": {"file": "/t"}})).unwrap(),
        false
    )]
    fn explicit_aws_is_chosen_as_python_chooses_it(
        #[case] info: ExternalAccountInfo,
        #[case] expected: bool,
    ) {
        assert_eq!(uses_explicit_aws(&info), expected);
    }

    #[rstest]
    #[case::missing(json!({"aws_access_key_id": "key"}))]
    #[case::empty(json!({"aws_access_key_id": "key", "aws_region_name": ""}))]
    #[case::null(json!({"aws_region_name": null}))]
    fn a_missing_region_is_refused_before_any_aws_call(#[case] extra: Value) {
        let Err(failure) = explicit_aws_credentials(&aws_info(extra), Arc::new(|_| None)) else {
            panic!("a credential without aws_region_name should not build");
        };
        assert_eq!(
            failure,
            GoogleLoginFailure::new(PythonException::Value, MISSING_REGION)
        );
    }

    #[rstest]
    fn the_subject_token_is_google_auth_s_serialized_signed_request() {
        let credentials = litellm_auth_aws::Credentials::new(
            "AKIDEXAMPLE",
            "secret",
            Some("session".into()),
            None,
            "test",
        );
        let token = caller_identity_subject_token(
            &credentials,
            "us-east-1",
            &CALLER_IDENTITY_URL.replace("{region}", "us-east-1"),
            "//iam.googleapis.com/projects/1/pool",
            UNIX_EPOCH + Duration::from_secs(1_700_000_000),
        )
        .unwrap();
        let decoded = percent_encoding::percent_decode_str(&token)
            .decode_utf8()
            .unwrap();
        let request: Value = serde_json::from_str(&decoded).unwrap();
        let keys: Vec<&str> = request["headers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|header| header["key"].as_str().unwrap())
            .collect();
        assert_eq!(
            keys,
            [
                "Authorization",
                "host",
                "x-amz-date",
                "x-amz-security-token",
                TARGET_RESOURCE_HEADER
            ]
        );
        assert_eq!(request["method"], "POST");
        assert_eq!(
            request["url"],
            "https://sts.us-east-1.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15"
        );
        assert_eq!(request["headers"][2]["value"], "20231114T221320Z");
        assert!(
            request["headers"][0]["value"]
                .as_str()
                .unwrap()
                .starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20231114/us-east-1/sts/aws4_request, SignedHeaders=host;x-amz-date;x-amz-security-token, ")
        );
        assert!(decoded.starts_with(r#"{"headers":[{"key":"Authorization","value":"#));
        assert!(!token.contains(':') && !token.contains('"'));
    }

    /// An EC2 metadata service and STS endpoint, recording each request as `METHOD path`
    /// with the session token it carried.
    async fn metadata() -> (
        String,
        Arc<std::sync::Mutex<Vec<String>>>,
        tokio::task::JoinHandle<()>,
    ) {
        use axum::{
            Router,
            routing::{get, post, put},
        };
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let record = |method: &'static str, path: &'static str, answer: &'static str| {
            let seen = seen.clone();
            move |headers: axum::http::HeaderMap| {
                let token = headers
                    .get("x-aws-ec2-metadata-token")
                    .map(|token| token.to_str().unwrap().to_owned());
                seen.lock()
                    .unwrap()
                    .push(format!("{method} {path} {}", token.unwrap_or_default()));
                async move { answer }
            }
        };
        let app = Router::new()
            .route(
                "/latest/api/token",
                put(record("PUT", "/latest/api/token", "imds-token")),
            )
            .route("/region", get(record("GET", "/region", "eu-west-1b")))
            .route("/creds", get(record("GET", "/creds", "fz-role")))
            .route(
                "/creds/fz-role",
                get(record(
                    "GET",
                    "/creds/fz-role",
                    r#"{"AccessKeyId": "AKID", "SecretAccessKey": "secret", "Token": "session"}"#,
                )),
            )
            .route(
                "/token",
                post({
                    let seen = seen.clone();
                    move |body: String| {
                        seen.lock().unwrap().push(format!("POST /token {body}"));
                        async { r#"{"access_token": "aws-token", "expires_in": 3600}"# }
                    }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (url, seen, server)
    }

    #[tokio::test]
    async fn a_metadata_account_reads_the_region_and_keys_from_imdsv2() {
        let (url, seen, server) = metadata().await;
        let info = aws_info(json!({
            "token_url": format!("{url}/token"),
            "credential_source": {
                "environment_id": "aws1", "region_url": format!("{url}/region"), "url": format!("{url}/creds"),
                "imdsv2_session_token_url": format!("{url}/latest/api/token"),
                "regional_cred_verification_url": CALLER_IDENTITY_URL,
            },
        }));
        let subject = MetadataCallerIdentity {
            region_url: Some(format!("{url}/region")),
            security_credentials_url: Some(format!("{url}/creds")),
            imdsv2_session_token_url: Some(format!("{url}/latest/api/token")),
            verification_url: CALLER_IDENTITY_URL.into(),
            audience: "aud".into(),
            os_env: Arc::new(|_| None),
            region: tokio::sync::OnceCell::new(),
        };
        let account = ExternalAccount::new(
            Settings::from_info(&info).unwrap(),
            Box::new(subject),
            Requests::with_environment(&[]),
        );

        assert_eq!(
            crate::vertex_messages::credentials::Credentials::refresh(&account)
                .await
                .unwrap()
                .token
                .unwrap()
                .expose(),
            "aws-token"
        );

        let seen = seen.lock().unwrap().clone();
        assert_eq!(
            seen[..5],
            [
                "PUT /latest/api/token ",
                "GET /region imds-token",
                "PUT /latest/api/token ",
                "GET /creds imds-token",
                "GET /creds/fz-role imds-token",
            ]
        );
        let subject_token = seen[5]
            .split("&subject_token=")
            .nth(1)
            .unwrap()
            .split('&')
            .next()
            .unwrap();
        let decoded = percent_encoding::percent_decode_str(subject_token)
            .decode_utf8()
            .unwrap();
        let decoded = percent_encoding::percent_decode_str(&decoded)
            .decode_utf8()
            .unwrap()
            .into_owned();
        assert!(decoded.contains("https://sts.eu-west-1.amazonaws.com?Action=GetCallerIdentity"));
        assert!(decoded.contains("Credential=AKID/"));
        server.abort();
    }

    #[rstest]
    #[case::no_version(json!({"environment_id": "aws", "regional_cred_verification_url": "u"}), PythonException::InvalidResource, "No valid AWS 'credential_source' provided")]
    #[case::no_url(json!({"environment_id": "aws1"}), PythonException::InvalidResource, "No valid AWS 'credential_source' provided")]
    #[case::version_two(json!({"environment_id": "aws2", "regional_cred_verification_url": "u"}), PythonException::InvalidValue, "aws version '2' is not supported in the current build.")]
    fn a_metadata_source_is_checked_as_google_auth_checks_it(
        #[case] source: Value,
        #[case] exception: PythonException,
        #[case] message: &str,
    ) {
        let info = aws_info(json!({ "credential_source": source }));
        assert_eq!(
            metadata_aws_credentials(&info, Arc::new(|_| None)).err(),
            Some(GoogleLoginFailure::new(exception, message))
        );
    }
}
