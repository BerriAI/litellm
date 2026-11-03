use litellm_core_utils::settings::Lookup;

// Every Redis setting litellm/_redis.py reads, with how Rust handles it and where else Python looks for it.
// Warn rows are read by Python and dropped here with a warning naming the consequence, never an error.
// environment marks the kwargs _get_redis_env_kwarg_mapping reads as REDIS_<KWARG>, and fallback lists the names
// _get_redis_client_logic consults with get_secret when the merged kwargs lack the setting
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Handling {
    Read,
    Warn,
}

pub(super) struct Setting {
    pub(super) kwarg: &'static str,
    pub(super) handling: Handling,
    pub(super) environment: bool,
    pub(super) fallback: &'static [&'static str],
}

impl Setting {
    const fn new(
        kwarg: &'static str,
        handling: Handling,
        environment: bool,
        fallback: &'static [&'static str],
    ) -> Self {
        Self {
            kwarg,
            handling,
            environment,
            fallback,
        }
    }

    pub(super) fn environment_name(&self) -> Option<String> {
        self.environment
            .then(|| format!("REDIS_{}", self.kwarg.to_ascii_uppercase()))
    }
}

pub(super) const SETTINGS: &[Setting] = &[
    Setting::new("aws_iam_auth", Handling::Read, true, &[]),
    Setting::new("aws_iam_cache_name", Handling::Read, true, &[]),
    Setting::new(
        "aws_iam_region",
        Handling::Read,
        true,
        &["AWS_REGION", "AWS_DEFAULT_REGION"],
    ),
    Setting::new("aws_iam_serverless", Handling::Read, true, &[]),
    Setting::new("aws_iam_user_name", Handling::Read, true, &[]),
    Setting::new(
        "azure_client_id",
        Handling::Read,
        true,
        &["AZURE_CLIENT_ID"],
    ),
    Setting::new(
        "azure_client_secret",
        Handling::Read,
        true,
        &["AZURE_CLIENT_SECRET"],
    ),
    Setting::new(
        "azure_redis_ad_token",
        Handling::Read,
        true,
        &["REDIS_AZURE_AD_TOKEN"],
    ),
    Setting::new(
        "azure_tenant_id",
        Handling::Read,
        true,
        &["AZURE_TENANT_ID"],
    ),
    Setting::new("cache", Handling::Warn, true, &[]),
    Setting::new("cache_config", Handling::Warn, true, &[]),
    Setting::new("charset", Handling::Warn, true, &[]),
    Setting::new("client_name", Handling::Warn, true, &[]),
    Setting::new("connection_pool", Handling::Warn, false, &[]),
    Setting::new("credential_provider", Handling::Read, false, &[]),
    Setting::new("db", Handling::Read, true, &[]),
    Setting::new("decode_responses", Handling::Warn, true, &[]),
    Setting::new("encoding", Handling::Warn, true, &[]),
    Setting::new("encoding_errors", Handling::Warn, true, &[]),
    Setting::new("errors", Handling::Warn, true, &[]),
    Setting::new("event_dispatcher", Handling::Warn, true, &[]),
    Setting::new(
        "gcp_service_account",
        Handling::Read,
        true,
        &["REDIS_GCP_SERVICE_ACCOUNT"],
    ),
    Setting::new(
        "gcp_ssl_ca_certs",
        Handling::Read,
        true,
        &["REDIS_GCP_SSL_CA_CERTS"],
    ),
    Setting::new("health_check_interval", Handling::Read, true, &[]),
    Setting::new("host", Handling::Read, true, &[]),
    Setting::new("lib_name", Handling::Warn, true, &[]),
    Setting::new("lib_version", Handling::Warn, true, &[]),
    Setting::new("max_connections", Handling::Warn, true, &[]),
    Setting::new("namespace", Handling::Read, false, &[]),
    Setting::new("password", Handling::Read, true, &[]),
    Setting::new("port", Handling::Read, true, &[]),
    Setting::new("protocol", Handling::Warn, true, &[]),
    Setting::new("redis_connect_func", Handling::Warn, true, &[]),
    Setting::new("retry", Handling::Warn, false, &[]),
    Setting::new("retry_on_error", Handling::Warn, true, &[]),
    Setting::new("retry_on_timeout", Handling::Warn, true, &[]),
    Setting::new(
        "sentinel_nodes",
        Handling::Read,
        false,
        &["REDIS_SENTINEL_NODES"],
    ),
    Setting::new(
        "sentinel_password",
        Handling::Read,
        false,
        &["REDIS_SENTINEL_PASSWORD"],
    ),
    Setting::new(
        "service_name",
        Handling::Read,
        false,
        &["REDIS_SERVICE_NAME"],
    ),
    Setting::new("single_connection_client", Handling::Warn, true, &[]),
    Setting::new("socket_connect_timeout", Handling::Read, true, &[]),
    Setting::new("socket_keepalive", Handling::Warn, true, &[]),
    Setting::new("socket_keepalive_options", Handling::Warn, true, &[]),
    Setting::new("socket_timeout", Handling::Read, true, &[]),
    Setting::new("ssl", Handling::Read, true, &[]),
    Setting::new("ssl_ca_certs", Handling::Read, true, &[]),
    Setting::new("ssl_ca_data", Handling::Read, true, &[]),
    Setting::new("ssl_ca_path", Handling::Read, true, &[]),
    Setting::new("ssl_cert_reqs", Handling::Read, true, &[]),
    Setting::new("ssl_certfile", Handling::Read, true, &[]),
    Setting::new("ssl_check_hostname", Handling::Read, true, &[]),
    Setting::new("ssl_ciphers", Handling::Warn, true, &[]),
    Setting::new("ssl_keyfile", Handling::Read, true, &[]),
    Setting::new("ssl_min_version", Handling::Warn, true, &[]),
    Setting::new("ssl_ocsp_context", Handling::Warn, true, &[]),
    Setting::new("ssl_ocsp_expected_cert", Handling::Warn, true, &[]),
    Setting::new("ssl_password", Handling::Warn, true, &[]),
    Setting::new("ssl_validate_ocsp", Handling::Warn, true, &[]),
    Setting::new("ssl_validate_ocsp_stapled", Handling::Warn, true, &[]),
    Setting::new(
        "startup_nodes",
        Handling::Read,
        false,
        &["REDIS_CLUSTER_NODES"],
    ),
    Setting::new("unix_socket_path", Handling::Read, true, &[]),
    Setting::new("url", Handling::Read, true, &[]),
    Setting::new("username", Handling::Read, true, &[]),
];

/// What ignoring a warned setting means for the connection, so the operator reads the consequence
/// rather than only the key name.
pub(super) fn consequence(kwarg: &str) -> &'static str {
    match kwarg {
        "ssl_validate_ocsp"
        | "ssl_validate_ocsp_stapled"
        | "ssl_ocsp_context"
        | "ssl_ocsp_expected_cert" => {
            "rustls has no client-side OCSP, so the connection is made without revocation checks"
        }
        "ssl_ciphers" | "ssl_min_version" => {
            "rustls negotiates TLS 1.2+ with its own ECDHE AEAD suites"
        }
        "ssl_password" => "an encrypted client key cannot be read, so the TLS handshake fails",
        "credential_provider" | "redis_connect_func" => {
            "the Python object is not callable from Rust, so static username and password are used"
        }
        "connection_pool" | "retry" | "event_dispatcher" => {
            "the Rust client uses its own pool, retries and events"
        }
        _ => "the Rust client has no equivalent",
    }
}

pub(super) fn setting(kwarg: &str) -> Option<&'static Setting> {
    SETTINGS.iter().find(|setting| setting.kwarg == kwarg)
}

pub(super) fn fallback(kwarg: &str, lookup: &dyn Lookup) -> Option<(&'static str, String)> {
    let names = setting(kwarg).map_or(&[][..], |setting| setting.fallback);
    let values: Vec<(&'static str, Option<String>)> =
        names.iter().map(|name| (*name, lookup.get(name))).collect();
    // Python chains these with `or`, so the first non-empty value wins and an empty one only when it is last
    values
        .iter()
        .find(|(_, value)| value.as_deref().is_some_and(|value| !value.is_empty()))
        .or(values.last())
        .and_then(|(name, value)| Some((*name, value.clone()?)))
}

pub(super) fn fallback_value(kwarg: &str, lookup: &dyn Lookup) -> Option<String> {
    fallback(kwarg, lookup).map(|(_, value)| value)
}
