//! LiteLLM's explicit AWS login for an external account, Python's
//! `VertexAIAwsWifAuth.credentials_from_explicit_aws`: a GetCallerIdentity request signed with the
//! AWS credentials LiteLLM resolves from the file's own `aws_*` keys, serialized as the subject
//! token Google's Rust SDK exchanges at STS.

use std::{sync::Arc, time::SystemTime};

use google_cloud_auth::credentials::{
    external_account::ProgrammaticBuilder,
    subject_token::{self, SubjectToken, SubjectTokenProvider},
};
use http::{HeaderMap, HeaderValue, header::AUTHORIZATION};
use litellm_auth_aws::{AwsAuthConfig, AwsAuthService, SigningScope, sign_headers};
use litellm_python_compat::urllib::quote;
use litellm_secret_value::SecretValue;
use serde::Deserialize;
use serde_json::json;

use super::{
    environment::GoogleEnvironment,
    failure::{GoogleLoginFailure, PythonException},
    source::{ExternalAccountInfo, present, target_principal},
};

/// LiteLLM's own `aws_*` keys in an external account file. Each is `Some(None)` when null, which
/// still chooses the explicit AWS login.
#[derive(Clone, Default, Deserialize, PartialEq, veil::Redact)]
pub(crate) struct ExplicitAwsKeys {
    #[serde(default, deserialize_with = "present")]
    aws_access_key_id: Option<Option<String>>,
    #[redact(with = "[REDACTED]")]
    #[serde(default, deserialize_with = "present")]
    aws_secret_access_key: Option<Option<String>>,
    #[redact(with = "[REDACTED]")]
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
    pub(crate) fn any_given(&self) -> bool {
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

/// Python's `_credentials_from_json` routes an `external_account` whose `environment_id` names
/// AWS and that carries any of LiteLLM's `aws_*` keys, even a null one, to
/// `VertexAIAwsWifAuth.credentials_from_explicit_aws`.
pub(crate) fn uses_explicit_aws(info: &ExternalAccountInfo) -> bool {
    info.credential_source
        .as_ref()
        .and_then(|source| source.environment_id.as_deref())
        .is_some_and(|environment_id| environment_id.contains("aws"))
        && info.aws.any_given()
}

/// The external account Google's Rust SDK exchanges the signed request for at STS, and then
/// for the service account its impersonation URL names.
pub(crate) fn explicit_aws_credentials(
    info: &ExternalAccountInfo,
    environment: Arc<dyn GoogleEnvironment>,
) -> Result<ProgrammaticBuilder, GoogleLoginFailure> {
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
    let audience = info.audience.clone().unwrap_or_default();
    let subject = SignedCallerIdentity {
        aws: AwsAuthService::default(),
        config,
        region,
        audience: audience.clone(),
        environment,
    };
    let builder = ProgrammaticBuilder::new(Arc::new(subject))
        .with_audience(audience)
        .with_subject_token_type(info.subject_token_type.clone().unwrap_or_default());
    let builder = match &info.token_url {
        Some(url) => builder.with_token_url(url),
        None => builder,
    };
    Ok(
        match info
            .service_account_impersonation_url
            .as_deref()
            .and_then(target_principal)
        {
            Some(principal) => builder.with_target_principal(principal),
            None => builder,
        },
    )
}

/// The subject token of LiteLLM's explicit AWS login: a GetCallerIdentity request signed with
/// the credentials LiteLLM's AWS login resolves from the file's `aws_*` keys.
struct SignedCallerIdentity {
    aws: AwsAuthService,
    config: AwsAuthConfig,
    region: String,
    audience: String,
    environment: Arc<dyn GoogleEnvironment>,
}

impl std::fmt::Debug for SignedCallerIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SignedCallerIdentity")
            .field("region", &self.region)
            .finish_non_exhaustive()
    }
}

fn refresh_error(error: impl std::fmt::Display) -> GoogleLoginFailure {
    GoogleLoginFailure::new(PythonException::Refresh, error.to_string())
}

impl SubjectTokenProvider for SignedCallerIdentity {
    type Error = GoogleLoginFailure;

    async fn subject_token(&self) -> Result<SubjectToken, GoogleLoginFailure> {
        let environment = self.environment.clone();
        let credentials = self
            .aws
            .resolve_credentials(self.config.clone(), &move |name| environment.os_env(name))
            .await
            .map_err(refresh_error)?;
        caller_identity_subject_token(
            &credentials,
            &self.region,
            &CALLER_IDENTITY_URL.replace("{region}", &self.region),
            &self.audience,
            SystemTime::now(),
        )
        .map(|token| subject_token::Builder::new(token).build())
        .map_err(refresh_error)
    }
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
        let environment = Arc::new(crate::vertex_messages::OnGce {
            project: "p",
            token: "t",
        });
        let Err(failure) = explicit_aws_credentials(&aws_info(extra), environment) else {
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
}
