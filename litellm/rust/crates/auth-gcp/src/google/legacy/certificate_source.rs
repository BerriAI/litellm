//! Backwards compatibility only, with Python google-auth's x509 workload identity federation: an
//! external account whose `credential_source` is a certificate. Google's Rust SDK
//! (`google_cloud_auth` 1.16) has no certificate credential source, so this module reads the
//! certificate configuration, sends the certificate chain as the subject token to the STS
//! endpoint over mutual TLS as google-auth's `identity_pool.py` does, and hands the STS token to
//! the SDK's impersonation when the file names a service account, or to [`super::impersonation`]
//! when that service account is impersonated elsewhere than the SDK's host. Delete it once the SDK
//! adds a certificate source.

use std::{path::PathBuf, sync::Arc};

use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_http::ClientIdentity;
use litellm_secret_value::SecretValue;
use rustls::pki_types::{CertificateDer, pem::PemObject};
use serde::Deserialize;

use super::{
    impersonation,
    token::{Compat, CompatLogin, Granted, HeldToken, exchange, impersonating},
};
use crate::google::{
    credentials::{CertificateSource, ExternalAccountInfo},
    environment::GoogleEnvironment,
    failure::{GoogleLoginFailure, PythonException, UNLOADABLE_CREDENTIALS},
};
use crate::{GcpTokenSource, constants::CLOUD_PLATFORM_SCOPE};

const TOKEN_EXCHANGE_GRANT: &str = "urn:ietf:params:oauth:grant-type:token-exchange";
const ACCESS_TOKEN_TYPE: &str = "urn:ietf:params:oauth:token-type:access_token";
const MTLS_TOKEN_URL: &str = "https://sts.mtls.googleapis.com/v1/token";
const CONFIG_ENV: &str = "GOOGLE_API_CERTIFICATE_CONFIG";

/// gcloud's `certificate_config.json`: where the workload's certificate and key are.
#[derive(Deserialize)]
struct CertificateConfig {
    cert_configs: CertConfigs,
}

#[derive(Deserialize)]
struct CertConfigs {
    workload: Workload,
}

#[derive(Deserialize)]
struct Workload {
    cert_path: PathBuf,
    key_path: PathBuf,
}

/// An x509 workload identity federation login: the certificate configuration it reads on each
/// exchange and the STS exchange it makes with the certificate.
pub(crate) struct CertificateLogin {
    config_path: PathBuf,
    trust_chain_path: Option<PathBuf>,
    token_url: String,
    audience: String,
    subject_token_type: String,
    client: Option<(String, String)>,
    held: HeldToken,
    environment: Arc<dyn GoogleEnvironment>,
}

fn malformed(message: &str) -> GoogleLoginFailure {
    GoogleLoginFailure::new(PythonException::Malformed, message)
}

fn refresh_error(message: String) -> GoogleLoginFailure {
    GoogleLoginFailure::new(PythonException::Refresh, message)
}

/// `identity_pool.Credentials` with a certificate source, impersonating the service account its
/// file names through Google's Rust SDK.
pub(crate) fn load(
    info: &ExternalAccountInfo,
    certificate: &CertificateSource,
    environment: Arc<dyn GoogleEnvironment>,
) -> Result<Arc<dyn GcpTokenSource>, GoogleLoginFailure> {
    let named = certificate
        .certificate_config_location
        .clone()
        .filter(|location| !location.is_empty());
    let config_path = match (named, certificate.use_default_certificate_config) {
        (Some(_), Some(true)) => {
            return Err(malformed(
                "certificate_config_location cannot be set when use_default_certificate_config is true.",
            ));
        }
        (None, None | Some(false)) => {
            return Err(malformed(
                "use_default_certificate_config must be true when no certificate_config_location is set.",
            ));
        }
        (Some(location), _) => PathBuf::from(location),
        (None, Some(true)) => environment
            .os_env(CONFIG_ENV)
            .filter(|path| !path.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(environment.os_env("HOME").unwrap_or_default())
                    .join(".config/gcloud/certificate_config.json")
            }),
    };
    let login = Arc::new(CertificateLogin {
        config_path,
        trust_chain_path: certificate
            .trust_chain_path
            .clone()
            .filter(|path| !path.is_empty())
            .map(PathBuf::from),
        token_url: info
            .token_url
            .clone()
            .unwrap_or_else(|| MTLS_TOKEN_URL.into()),
        audience: info.audience.clone().unwrap_or_default(),
        subject_token_type: info.subject_token_type.clone().unwrap_or_default(),
        client: info
            .client_id
            .clone()
            .map(|id| (id, info.client_secret.clone().unwrap_or_default())),
        held: HeldToken::default(),
        environment: environment.clone(),
    });
    let lifetime = info
        .service_account_impersonation
        .as_ref()
        .and_then(|options| options.token_lifetime_seconds);
    match &info.service_account_impersonation_url {
        Some(url) if !impersonation::sdk_calls(url) => Ok(super::impersonated_at(
            Arc::new(Compat(login)),
            url,
            lifetime,
            environment,
        )),
        Some(url) => impersonating(login, url, lifetime)?.build().map_err(|_| {
            GoogleLoginFailure::new(PythonException::InvalidValue, UNLOADABLE_CREDENTIALS)
        }),
        None => Ok(Arc::new(Compat(login))),
    }
}

/// The certificates a PEM file holds, in order.
fn certificates(path: &PathBuf) -> Result<Vec<CertificateDer<'static>>, GoogleLoginFailure> {
    let unreadable = |error: &dyn std::fmt::Display| {
        refresh_error(format!(
            "Cannot read the certificates in {}: {error}",
            path.display()
        ))
    };
    CertificateDer::pem_file_iter(path)
        .map_err(|error| unreadable(&error))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| unreadable(&error))
}

impl CertificateLogin {
    fn workload(&self) -> Result<Workload, GoogleLoginFailure> {
        let text = std::fs::read_to_string(&self.config_path).map_err(|error| {
            refresh_error(format!(
                "Cannot read the certificate configuration {}: {error}",
                self.config_path.display()
            ))
        })?;
        serde_json::from_str::<CertificateConfig>(&text)
            .map(|config| config.cert_configs.workload)
            .map_err(|error| {
                refresh_error(format!(
                    "The certificate configuration {} names no workload certificate: {error}",
                    self.config_path.display()
                ))
            })
    }

    /// `_X509Supplier.get_subject_token`: the leaf certificate, then the trust chain without it,
    /// each base64-encoded DER in a JSON array.
    fn subject_token(&self, workload: &Workload) -> Result<String, GoogleLoginFailure> {
        let leaf = certificates(&workload.cert_path)?
            .into_iter()
            .next()
            .ok_or_else(|| {
                refresh_error(format!(
                    "{} holds no certificate",
                    workload.cert_path.display()
                ))
            })?;
        let chain = match &self.trust_chain_path {
            Some(path) => certificates(path)?,
            None => Vec::new(),
        };
        let encoded: Vec<String> = std::iter::once(leaf.clone())
            .chain(chain.into_iter().filter(|certificate| *certificate != leaf))
            .map(|certificate| STANDARD.encode(certificate))
            .collect();
        serde_json::to_string(&encoded).map_err(|error| refresh_error(error.to_string()))
    }

    /// `sts.Client.exchange_token` over mutual TLS with the workload's certificate and key.
    async fn exchange(&self) -> Result<Granted, GoogleLoginFailure> {
        let workload = self.workload()?;
        let subject_token = self.subject_token(&workload)?;
        let request = self
            .environment
            .http_client(
                &self.token_url,
                Some(ClientIdentity::Split {
                    certificate: workload.cert_path,
                    key: workload.key_path,
                }),
            )?
            .post(&self.token_url)
            .form(&[
                ("grant_type", TOKEN_EXCHANGE_GRANT),
                ("audience", &self.audience),
                ("scope", CLOUD_PLATFORM_SCOPE),
                ("requested_token_type", ACCESS_TOKEN_TYPE),
                ("subject_token", &subject_token),
                ("subject_token_type", &self.subject_token_type),
            ]);
        exchange(match &self.client {
            Some((id, secret)) => request.basic_auth(id, Some(secret)),
            None => request,
        })
        .await
    }
}

impl CompatLogin for CertificateLogin {
    async fn token(&self) -> Result<SecretValue, GoogleLoginFailure> {
        self.held.get_or_fetch(self.exchange()).await
    }
}
