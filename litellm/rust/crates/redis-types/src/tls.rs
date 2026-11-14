use std::path::PathBuf;

use litellm_secret_value::SecretValue;

#[derive(Clone, Debug, PartialEq)]
pub struct TlsOptions {
    pub ca_certs: Option<PathBuf>,
    /// PEM text trusted alongside `ca_certs`, redis-py's `ssl_ca_data`
    pub ca_data: Option<String>,
    /// A directory whose files are PEM certificates to trust, redis-py's `ssl_ca_path`
    pub ca_path: Option<PathBuf>,
    pub certfile: Option<PathBuf>,
    pub keyfile: Option<PathBuf>,
    /// Unlocks an encrypted `keyfile`, redis-py's `ssl_password`
    pub key_password: Option<SecretValue>,
    pub verify_certificate: bool,
    pub check_hostname: bool,
}

impl TlsOptions {
    pub fn verified() -> Self {
        Self {
            ca_certs: None,
            ca_data: None,
            ca_path: None,
            certfile: None,
            keyfile: None,
            key_password: None,
            verify_certificate: true,
            check_hostname: false,
        }
    }
}
