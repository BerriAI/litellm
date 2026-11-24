//! The credential files google-auth reads, parsed once into typed credentials, and size-capped
//! reads of the files they name.

use std::io::Read;

use serde::{Deserialize, Deserializer};
use serde_json::{Map, Value};

use super::legacy::ExternalAuthorizedUserInfo;
use crate::identity::SdkJson;

/// The most characters any credentials file holds.
pub(crate) const MAX_CREDENTIALS_FILE_CHARACTERS: usize = 1 << 20;
/// The most bytes any credentials file is read to: four UTF-8 bytes for each of
/// [`MAX_CREDENTIALS_FILE_CHARACTERS`].
pub(crate) const MAX_CREDENTIALS_FILE_BYTES: u64 = 4 * MAX_CREDENTIALS_FILE_CHARACTERS as u64;

/// A service account key, as `from_service_account_info` reads it, and the key itself for
/// Google's Rust SDK.
#[derive(Clone, Default, Deserialize, PartialEq, veil::Redact)]
pub(crate) struct ServiceAccountInfo {
    pub(crate) client_email: Option<String>,
    pub(crate) token_uri: Option<String>,
    #[redact(with = "[REDACTED]")]
    pub(crate) private_key: Option<String>,
    pub(crate) private_key_id: Option<String>,
    pub(crate) project_id: Option<String>,
    #[serde(skip)]
    pub(crate) sdk: SdkJson,
}

/// An authorized user, as `Credentials.from_authorized_user_info` reads it, and the file itself
/// for Google's Rust SDK.
#[derive(Clone, Default, Deserialize, PartialEq, veil::Redact)]
pub(crate) struct AuthorizedUserInfo {
    #[redact(with = "[REDACTED]")]
    pub(crate) refresh_token: Option<String>,
    pub(crate) client_id: Option<String>,
    #[redact(with = "[REDACTED]")]
    pub(crate) client_secret: Option<String>,
    #[redact(with = "[REDACTED]")]
    pub(crate) rapt_token: Option<String>,
    pub(crate) universe_domain: Option<String>,
    pub(crate) quota_project_id: Option<String>,
    #[serde(skip)]
    pub(crate) sdk: SdkJson,
}

/// An external account's file, as `external_account.Credentials.from_info` and LiteLLM's
/// explicit AWS login read it, and the file itself for Google's Rust SDK.
#[derive(Clone, Default, Deserialize, PartialEq, veil::Redact)]
pub(crate) struct ExternalAccountInfo {
    pub(crate) audience: Option<String>,
    pub(crate) subject_token_type: Option<String>,
    pub(crate) token_url: Option<String>,
    pub(crate) service_account_impersonation_url: Option<String>,
    pub(crate) service_account_impersonation: Option<ImpersonationOptions>,
    pub(crate) client_id: Option<String>,
    #[redact(with = "[REDACTED]")]
    pub(crate) client_secret: Option<String>,
    pub(crate) workforce_pool_user_project: Option<String>,
    pub(crate) universe_domain: Option<String>,
    pub(crate) credential_source: Option<CredentialSource>,
    #[serde(flatten)]
    pub(crate) aws: ExplicitAwsKeys,
    #[serde(skip)]
    pub(crate) sdk: SdkJson,
}

/// `service_account_impersonation`'s options.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub(crate) struct ImpersonationOptions {
    pub(crate) token_lifetime_seconds: Option<u64>,
}

/// Where an external account reads its subject token, as far as the login routes it: AWS, a URL,
/// an executable, a certificate or a file. Google's Rust SDK reads the rest of it.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub(crate) struct CredentialSource {
    pub(crate) environment_id: Option<String>,
    pub(crate) url: Option<serde::de::IgnoredAny>,
    pub(crate) executable: Option<ExecutableSource>,
    pub(crate) certificate: Option<CertificateSource>,
    pub(crate) file: Option<String>,
    pub(crate) format: Option<SourceFormat>,
}

/// How a file source holds its subject token: the whole file, or one field of a JSON object.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub(crate) struct SourceFormat {
    #[serde(rename = "type")]
    pub(crate) kind: Option<String>,
    pub(crate) subject_token_field_name: Option<String>,
}

/// A credential helper executable: its command line, how long it may run and where it caches its
/// answer.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub(crate) struct ExecutableSource {
    pub(crate) command: Option<String>,
    pub(crate) timeout_millis: Option<u64>,
    pub(crate) interactive_timeout_millis: Option<u64>,
    pub(crate) output_file: Option<String>,
}

/// An x509 workload identity federation source: where its certificate configuration and trust
/// chain are.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub(crate) struct CertificateSource {
    pub(crate) certificate_config_location: Option<String>,
    pub(crate) use_default_certificate_config: Option<bool>,
    pub(crate) trust_chain_path: Option<String>,
}

/// The impersonated service account credentials an application default credentials file holds:
/// the fields the login checks, and the file itself for Google's Rust SDK to build them from.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub(crate) struct ImpersonatedInfo {
    pub(crate) source_credentials: Option<SourceCredentials>,
    pub(crate) service_account_impersonation_url: Option<String>,
    #[serde(skip)]
    pub(crate) sdk: SdkJson,
}

/// The credentials an impersonated service account is minted from, by type.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub(crate) struct SourceCredentials {
    #[serde(rename = "type")]
    pub(crate) kind: Option<String>,
    /// The source when it is a workforce identity user, which Google's Rust SDK cannot build.
    #[serde(flatten)]
    pub(crate) external_authorized_user: ExternalAuthorizedUserInfo,
}

/// The `type` a credentials file declares, as google-auth spells it.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::IntoStaticStr, strum::VariantArray,
)]
pub(crate) enum CredentialsType {
    #[strum(serialize = "authorized_user")]
    AuthorizedUser,
    #[strum(serialize = "service_account")]
    ServiceAccount,
    #[strum(serialize = "external_account")]
    ExternalAccount,
    #[strum(serialize = "external_account_authorized_user")]
    ExternalAccountAuthorizedUser,
    #[strum(serialize = "impersonated_service_account")]
    ImpersonatedServiceAccount,
    #[strum(serialize = "gdch_service_account")]
    GdchServiceAccount,
}

impl CredentialsType {
    /// The type `info` declares, `None` when it declares none or one google-auth does not know.
    pub(crate) fn declared_by(info: &Map<String, Value>) -> Option<Self> {
        Self::named(info.get("type").and_then(Value::as_str))
    }

    /// The type `kind` names, `None` when it is absent or not one google-auth knows.
    pub(crate) fn named(kind: Option<&str>) -> Option<Self> {
        kind.and_then(|kind| kind.parse().ok())
    }
}

/// A credentials file, by the `type` it declares.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub(crate) enum CredentialsFile {
    #[serde(rename = "service_account")]
    ServiceAccount(ServiceAccountInfo),
    #[serde(rename = "authorized_user")]
    AuthorizedUser(AuthorizedUserInfo),
    #[serde(rename = "external_account")]
    ExternalAccount(Box<ExternalAccountInfo>),
    #[serde(rename = "external_account_authorized_user")]
    ExternalAccountAuthorizedUser(ExternalAuthorizedUserInfo),
    #[serde(rename = "impersonated_service_account")]
    ImpersonatedServiceAccount(ImpersonatedInfo),
    #[serde(rename = "gdch_service_account")]
    GdchServiceAccount,
    #[serde(other)]
    Unknown,
}

/// Why a credentials JSON object could not be read as the credentials its type names.
#[derive(Debug)]
pub(crate) enum Unparsed {
    /// The object declares no `type`, or one that is not text.
    Untyped,
    /// A field holds a value of a type the credentials cannot hold.
    Field(serde_json::Error),
}

impl CredentialsFile {
    /// The credentials an object holds, by the `type` it declares.
    pub(crate) fn parse(info: Map<String, Value>) -> Result<Self, Unparsed> {
        if !info.get("type").is_some_and(Value::is_string) {
            return Err(Unparsed::Untyped);
        }
        let parsed =
            serde_json::from_value::<Self>(Value::Object(info.clone())).map_err(Unparsed::Field)?;
        Ok(match parsed {
            Self::ServiceAccount(account) => Self::ServiceAccount(ServiceAccountInfo {
                sdk: SdkJson(info),
                ..account
            }),
            Self::AuthorizedUser(user) => Self::AuthorizedUser(AuthorizedUserInfo {
                sdk: SdkJson(info),
                ..user
            }),
            Self::ExternalAccount(account) => {
                Self::ExternalAccount(Box::new(ExternalAccountInfo {
                    sdk: SdkJson(info),
                    ..*account
                }))
            }
            Self::ImpersonatedServiceAccount(account) => {
                Self::ImpersonatedServiceAccount(ImpersonatedInfo {
                    sdk: SdkJson(info),
                    ..account
                })
            }
            other => other,
        })
    }
}

/// LiteLLM's own `aws_*` keys in an external account file. Each is `Some(None)` when null, which
/// still chooses the explicit AWS login.
#[derive(Clone, Default, Deserialize, PartialEq, veil::Redact)]
pub(crate) struct ExplicitAwsKeys {
    #[serde(default, deserialize_with = "present")]
    pub(crate) aws_access_key_id: Option<Option<String>>,
    #[redact(with = "[REDACTED]")]
    #[serde(default, deserialize_with = "present")]
    pub(crate) aws_secret_access_key: Option<Option<String>>,
    #[redact(with = "[REDACTED]")]
    #[serde(default, deserialize_with = "present")]
    pub(crate) aws_session_token: Option<Option<String>>,
    #[serde(default, deserialize_with = "present")]
    pub(crate) aws_region_name: Option<Option<String>>,
    #[serde(default, deserialize_with = "present")]
    pub(crate) aws_session_name: Option<Option<String>>,
    #[serde(default, deserialize_with = "present")]
    pub(crate) aws_profile_name: Option<Option<String>>,
    #[serde(default, deserialize_with = "present")]
    pub(crate) aws_role_name: Option<Option<String>>,
    #[serde(default, deserialize_with = "present")]
    pub(crate) aws_web_identity_token: Option<Option<String>>,
    #[serde(default, deserialize_with = "present")]
    pub(crate) aws_sts_endpoint: Option<Option<String>>,
    #[serde(default, deserialize_with = "present")]
    pub(crate) aws_external_id: Option<Option<String>>,
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

/// `IAMCredentials`' `service_account_email`: the text between the last `/` of an
/// impersonation URL and `:generateAccessToken`.
pub(crate) fn target_principal(url: &str) -> Option<&str> {
    url.rsplit_once('/')?
        .1
        .strip_suffix(":generateAccessToken")
        .filter(|principal| !principal.is_empty())
}

/// `#[serde(deserialize_with)]` for a field that may be absent, null or a value: `None` when it
/// is absent and `Some(None)` when it is null.
pub(crate) fn present<'de, D, T>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer).map(Some)
}

/// Why [`read_bounded`] gave no contents.
#[derive(Debug)]
pub(crate) enum BoundedRead {
    /// The file could not be opened or read.
    Unreadable(std::io::Error),
    /// The file holds more than [`MAX_CREDENTIALS_FILE_BYTES`].
    TooLarge,
}

/// A file's bytes, read no further than one byte past [`MAX_CREDENTIALS_FILE_BYTES`], so a file
/// such as `/dev/zero` is refused by its size instead of read without end.
pub(crate) fn read_bounded(path: &str) -> Result<Vec<u8>, BoundedRead> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .and_then(|file| {
            file.take(MAX_CREDENTIALS_FILE_BYTES + 1)
                .read_to_end(&mut bytes)
        })
        .map_err(BoundedRead::Unreadable)?;
    match bytes.len() as u64 > MAX_CREDENTIALS_FILE_BYTES {
        true => Err(BoundedRead::TooLarge),
        false => Ok(bytes),
    }
}

/// [`read_bounded`] on a blocking thread, for a file read while a token is fetched. A read that
/// did not finish is unreadable.
pub(crate) async fn read_bounded_off_runtime(path: String) -> Result<Vec<u8>, BoundedRead> {
    tokio::task::spawn_blocking(move || read_bounded(&path))
        .await
        .unwrap_or_else(|error| Err(BoundedRead::Unreadable(std::io::Error::other(error))))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::authorized_user(json!({"type": "authorized_user"}), Some(CredentialsType::AuthorizedUser))]
    #[case::impersonated(json!({"type": "impersonated_service_account"}), Some(CredentialsType::ImpersonatedServiceAccount))]
    #[case::unknown(json!({"type": "something_else"}), None)]
    #[case::not_text(json!({"type": 1}), None)]
    #[case::untyped(json!({}), None)]
    fn the_declared_type_is_read_by_its_google_auth_spelling(
        #[case] info: Value,
        #[case] expected: Option<CredentialsType>,
    ) {
        let Value::Object(info) = info else {
            unreachable!("every case is an object")
        };
        assert_eq!(CredentialsType::declared_by(&info), expected);
    }

    #[rstest]
    fn credentials_file_json_debug_never_shows_a_secret() {
        let Value::Object(info) = json!({"private_key": "credential-value"}) else {
            unreachable!()
        };
        let debug = format!("{:?}", SdkJson(info));
        assert!(!debug.contains("credential-value"), "{debug}");
        assert!(debug.contains("[REDACTED]"), "{debug}");
    }

    #[rstest]
    #[case::service_account(json!({"type": "service_account"}), true)]
    #[case::external_account(json!({"type": "external_account", "audience": "a"}), true)]
    #[case::gdch(json!({"type": "gdch_service_account", "anything": 1}), true)]
    #[case::unknown(json!({"type": "something_else"}), true)]
    #[case::untyped(json!({"project_id": "p"}), false)]
    #[case::numeric_type(json!({"type": 5}), false)]
    fn a_file_is_read_by_the_type_it_declares(#[case] info: Value, #[case] typed: bool) {
        let parsed = CredentialsFile::parse(info.as_object().unwrap().clone());
        assert_eq!(parsed.is_ok(), typed, "{parsed:?}");
    }

    #[rstest]
    fn an_external_account_keeps_its_file_for_google_s_sdk() {
        let info = json!({"type": "external_account", "audience": "a", "credential_source": {"certificate": {}}});
        let Ok(CredentialsFile::ExternalAccount(account)) =
            CredentialsFile::parse(info.as_object().unwrap().clone())
        else {
            panic!("not an external account");
        };
        assert_eq!(account.audience.as_deref(), Some("a"));
        assert_eq!(account.sdk, SdkJson(info.as_object().unwrap().clone()));
    }
}
