use std::{path::PathBuf, time::Duration};

use litellm_core_utils::settings::Lookup;
use litellm_secrets_types::SecretValue;

use crate::Error;

const DEFAULT_ADDRESS: &str = "http://127.0.0.1:8200";
const DEFAULT_MOUNT: &str = "secret";
const DEFAULT_APPROLE_MOUNT_PATH: &str = "approle";
const DEFAULT_REFRESH_INTERVAL: Duration = Duration::from_secs(86400);
const HCP_VAULT_ADDR: &str = "HCP_VAULT_ADDR";
const HCP_VAULT_TOKEN: &str = "HCP_VAULT_TOKEN";
const HCP_VAULT_NAMESPACE: &str = "HCP_VAULT_NAMESPACE";
const HCP_VAULT_LOGIN_NAMESPACE: &str = "HCP_VAULT_LOGIN_NAMESPACE";
const HCP_VAULT_SECRET_NAMESPACE: &str = "HCP_VAULT_SECRET_NAMESPACE";
const HCP_VAULT_MOUNT_NAME: &str = "HCP_VAULT_MOUNT_NAME";
const HCP_VAULT_PATH_PREFIX: &str = "HCP_VAULT_PATH_PREFIX";
const HCP_VAULT_APPROLE_ROLE_ID: &str = "HCP_VAULT_APPROLE_ROLE_ID";
const HCP_VAULT_APPROLE_SECRET_ID: &str = "HCP_VAULT_APPROLE_SECRET_ID";
const HCP_VAULT_APPROLE_MOUNT_PATH: &str = "HCP_VAULT_APPROLE_MOUNT_PATH";
const HCP_VAULT_CLIENT_CERT: &str = "HCP_VAULT_CLIENT_CERT";
const HCP_VAULT_CLIENT_KEY: &str = "HCP_VAULT_CLIENT_KEY";
const HCP_VAULT_CERT_ROLE: &str = "HCP_VAULT_CERT_ROLE";
const HCP_VAULT_REFRESH_INTERVAL: &str = "HCP_VAULT_REFRESH_INTERVAL";
const SECRET_MANAGER_REFRESH_INTERVAL: &str = "SECRET_MANAGER_REFRESH_INTERVAL";

#[derive(Clone, Debug)]
pub struct AppRoleAuth {
    pub role_id: String,
    pub secret_id: SecretValue,
    pub mount_path: String,
}

#[derive(Clone, Debug)]
pub struct TlsCertAuth {
    pub cert_path: PathBuf,
    pub key_path: PathBuf,
    pub role: Option<String>,
}

#[derive(Clone, Debug)]
pub struct HashicorpVaultConfig {
    pub address: String,
    pub token: Option<SecretValue>,
    pub namespace: Option<String>,
    pub login_namespace: Option<String>,
    pub secret_namespace: Option<String>,
    pub mount: String,
    pub path_prefix: Option<String>,
    pub approle: Option<AppRoleAuth>,
    pub tls_cert: Option<TlsCertAuth>,
    pub refresh_interval: Duration,
}

impl HashicorpVaultConfig {
    pub fn from_environment(environment: &dyn Lookup) -> Result<Self, Error> {
        let address: String = environment
            .get(HCP_VAULT_ADDR)
            .and_then(|value| nonempty(value.trim()))
            .map(|value| value.trim_end_matches('/').to_owned())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| DEFAULT_ADDRESS.to_owned());
        let token: Option<SecretValue> = environment
            .get(HCP_VAULT_TOKEN)
            .and_then(nonempty)
            .map(SecretValue::new);
        let namespace: Option<String> = path_component(environment.get(HCP_VAULT_NAMESPACE));
        let login_namespace: Option<String> =
            path_component(environment.get(HCP_VAULT_LOGIN_NAMESPACE));
        let secret_namespace: Option<String> =
            path_component(environment.get(HCP_VAULT_SECRET_NAMESPACE));
        let mount: String = path_component(environment.get(HCP_VAULT_MOUNT_NAME))
            .unwrap_or_else(|| DEFAULT_MOUNT.to_owned());
        let path_prefix: Option<String> = path_component(environment.get(HCP_VAULT_PATH_PREFIX));
        let approle: Option<AppRoleAuth> = match (
            environment
                .get(HCP_VAULT_APPROLE_ROLE_ID)
                .and_then(nonempty),
            environment
                .get(HCP_VAULT_APPROLE_SECRET_ID)
                .and_then(nonempty)
                .map(SecretValue::new),
        ) {
            (Some(role_id), Some(secret_id)) => Some(AppRoleAuth {
                role_id,
                secret_id,
                mount_path: path_component(environment.get(HCP_VAULT_APPROLE_MOUNT_PATH))
                    .unwrap_or_else(|| DEFAULT_APPROLE_MOUNT_PATH.to_owned()),
            }),
            _ => None,
        };
        let tls_cert: Option<TlsCertAuth> = match (
            environment.get(HCP_VAULT_CLIENT_CERT).and_then(nonempty),
            environment.get(HCP_VAULT_CLIENT_KEY).and_then(nonempty),
        ) {
            (Some(cert_path), Some(key_path)) => Some(TlsCertAuth {
                cert_path: PathBuf::from(cert_path),
                key_path: PathBuf::from(key_path),
                role: environment.get(HCP_VAULT_CERT_ROLE).and_then(nonempty),
            }),
            _ => None,
        };
        let refresh_interval: Duration = refresh_interval(environment)?;
        Ok(Self {
            address,
            token,
            namespace,
            login_namespace,
            secret_namespace,
            mount,
            path_prefix,
            approle,
            tls_cert,
            refresh_interval,
        })
    }

    pub fn login_namespace(&self) -> Option<&str> {
        self.login_namespace
            .as_deref()
            .or(self.namespace.as_deref())
    }

    pub fn secret_namespace(&self) -> Option<&str> {
        self.secret_namespace
            .as_deref()
            .or(self.namespace.as_deref())
    }
}

fn nonempty(value: impl AsRef<str>) -> Option<String> {
    let value: &str = value.as_ref();
    (!value.is_empty()).then(|| value.to_owned())
}

fn path_component(value: Option<String>) -> Option<String> {
    value
        .and_then(|value| nonempty(value.trim()))
        .map(|value| value.trim_matches('/').to_owned())
        .filter(|value| !value.is_empty())
}

fn refresh_interval(environment: &dyn Lookup) -> Result<Duration, Error> {
    let value: Option<String> = environment
        .get(HCP_VAULT_REFRESH_INTERVAL)
        .and_then(nonempty)
        .or_else(|| {
            environment
                .get(SECRET_MANAGER_REFRESH_INTERVAL)
                .and_then(nonempty)
        });
    let Some(value) = value else {
        return Ok(DEFAULT_REFRESH_INTERVAL);
    };
    let seconds: i64 = value.parse().map_err(|_| Error::RefreshInterval)?;
    if seconds < 0 {
        return Err(Error::RefreshInterval);
    }
    Ok(Duration::from_secs(seconds as u64))
}
