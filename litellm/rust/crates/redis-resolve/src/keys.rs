use litellm_redis_types::Ignored;
use serde_json::Value;

/// How the Rust client honors a redis-py kwarg.
#[derive(Clone, Copy, Debug)]
pub enum Handling {
    /// Modeled by the resolver into a field of `Options`.
    Read,
    /// Dropped and returned as an `Ignored` naming `consequence`, unless `met` says the value asks for nothing the
    /// Rust client does not already do.
    Warn {
        consequence: &'static str,
        met: Option<fn(&Value) -> bool>,
    },
    /// A security check the Rust client cannot perform and Python's async client already refuses, so asking for it
    /// fails instead of connecting without it.
    Reject { check: &'static str, asked: Asked },
}

/// Which values of a `Reject` row ask for its check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Asked {
    /// A flag read as true. False asks for nothing.
    WhenTrue,
    /// Any present value, like a list of flags. An empty one asks for nothing.
    WhenSet,
}

/// Whether redis-py's `parse_url` reads the setting from a url's query and the Rust resolver does too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FromUrl {
    No,
    Any,
    Rediss,
}

#[derive(Clone, Copy, Debug)]
pub struct Setting {
    pub kwarg: &'static str,
    pub handling: Handling,
    pub from_url: FromUrl,
    /// `_get_redis_env_kwarg_mapping` reads it as `REDIS_<KWARG>`.
    pub environment: bool,
    /// The names `_merge_redis_kwargs` reads with `get_secret` when the merged kwargs lack the setting.
    pub fallback: &'static [&'static str],
}

impl Setting {
    const fn new(kwarg: &'static str, handling: Handling) -> Self {
        Self {
            kwarg,
            handling,
            from_url: FromUrl::No,
            environment: false,
            fallback: &[],
        }
    }

    const fn env(self) -> Self {
        Self {
            environment: true,
            ..self
        }
    }

    const fn in_url(self, from_url: FromUrl) -> Self {
        Self { from_url, ..self }
    }

    const fn or_secrets(self, fallback: &'static [&'static str]) -> Self {
        Self { fallback, ..self }
    }

    pub fn environment_name(&self) -> Option<String> {
        self.environment
            .then(|| format!("REDIS_{}", self.kwarg.to_ascii_uppercase()))
    }

    pub fn is_read(&self) -> bool {
        matches!(self.handling, Handling::Read)
    }

    /// The warning a set value earns, or `None` when the setting is read, refused or already met.
    pub(crate) fn warning(&self, value: &Value) -> Option<&'static str> {
        match self.handling {
            Handling::Warn { consequence, met } if !met.is_some_and(|met| met(value)) => {
                Some(consequence)
            }
            _ => None,
        }
    }
}

const fn read(kwarg: &'static str) -> Setting {
    Setting::new(kwarg, Handling::Read)
}

const fn warn(kwarg: &'static str, consequence: &'static str) -> Setting {
    Setting::new(
        kwarg,
        Handling::Warn {
            consequence,
            met: None,
        },
    )
}

const fn warn_unless(
    kwarg: &'static str,
    consequence: &'static str,
    met: fn(&Value) -> bool,
) -> Setting {
    Setting::new(
        kwarg,
        Handling::Warn {
            consequence,
            met: Some(met),
        },
    )
}

const fn reject(kwarg: &'static str, check: &'static str, asked: Asked) -> Setting {
    Setting::new(kwarg, Handling::Reject { check, asked })
}

const NO_EQUIVALENT: &str = "the Rust client has no equivalent";
const OWN_STACK: &str = "the Rust client uses its own pool, retries and events";
const OWN_CONNECTION: &str = "it tunes redis-py's own connection, and the Rust client has its own";
const OWN_CLUSTER: &str = "the Rust cluster client discovers, routes and retries on its own";
const MAINTENANCE: &str = "the Rust client does not follow server maintenance notifications";
const RUSTLS: &str = "rustls negotiates TLS 1.2+ with its own ECDHE AEAD suites";
const REVOCATION: &str = "OCSP revocation checks";
const OCSP: &str = "it only configures ssl_validate_ocsp, which the Rust client refuses when set";
const VERIFY_FLAGS: &str = "X.509 verify flags such as CRL checking";

/// Every Redis kwarg redis-py or `litellm/_redis.py` accepts, one row each. `redis-py-kwargs.json` records which
/// redis-py versions and `REDIS_*` names accept them, and `tests/catalog.rs` keeps the two in step.
pub const SETTINGS: &[Setting] = &[
    warn("address_remap", OWN_CLUSTER),
    warn("auto_close_connection_pool", OWN_STACK),
    read("aws_iam_auth").env(),
    read("aws_iam_cache_name").env(),
    read("aws_iam_region")
        .env()
        .or_secrets(&["AWS_REGION", "AWS_DEFAULT_REGION"]),
    read("aws_iam_serverless").env(),
    read("aws_iam_user_name").env(),
    read("azure_client_id")
        .env()
        .or_secrets(&["AZURE_CLIENT_ID"]),
    read("azure_client_secret")
        .env()
        .or_secrets(&["AZURE_CLIENT_SECRET"]),
    read("azure_redis_ad_token")
        .env()
        .or_secrets(&["REDIS_AZURE_AD_TOKEN"]),
    read("azure_tenant_id")
        .env()
        .or_secrets(&["AZURE_TENANT_ID"]),
    warn("cache", NO_EQUIVALENT).env(),
    warn("cache_config", NO_EQUIVALENT).env(),
    warn("charset", NO_EQUIVALENT).env(),
    warn("client_name", NO_EQUIVALENT).env(),
    warn("cluster_error_retry_attempts", OWN_CLUSTER),
    warn("command_packer", OWN_CONNECTION),
    warn("connection_class", OWN_STACK),
    warn("connection_error_retry_attempts", OWN_CLUSTER),
    warn("connection_pool", OWN_STACK),
    read("credential_provider"),
    read("db").env().in_url(FromUrl::Any),
    warn("decode_responses", NO_EQUIVALENT).env(),
    warn("driver_info", NO_EQUIVALENT),
    warn("dynamic_startup_nodes", OWN_CLUSTER),
    warn("encoder_class", OWN_CONNECTION),
    warn("encoding", NO_EQUIVALENT).env(),
    warn("encoding_errors", NO_EQUIVALENT).env(),
    warn("errors", NO_EQUIVALENT).env(),
    warn("event_dispatcher", OWN_STACK).env(),
    read("gcp_service_account")
        .env()
        .or_secrets(&["REDIS_GCP_SERVICE_ACCOUNT"]),
    read("gcp_ssl_ca_certs")
        .env()
        .or_secrets(&["REDIS_GCP_SSL_CA_CERTS"]),
    read("health_check_interval").env().in_url(FromUrl::Any),
    read("host").env(),
    warn("legacy_responses", NO_EQUIVALENT),
    warn("lib_name", NO_EQUIVALENT).env(),
    warn("lib_version", NO_EQUIVALENT).env(),
    warn("load_balancing_strategy", OWN_CLUSTER),
    warn("maint_notifications_config", MAINTENANCE),
    warn("maint_notifications_pool_handler", MAINTENANCE),
    warn("maintenance_notification_hash", MAINTENANCE),
    warn("maintenance_state", MAINTENANCE),
    warn("max_connections", "the Rust client has no connection cap").env(),
    read("namespace"),
    warn("orig_host_address", MAINTENANCE),
    warn("orig_socket_connect_timeout", MAINTENANCE),
    warn("orig_socket_timeout", MAINTENANCE),
    warn("oss_cluster_maint_notifications_handler", MAINTENANCE),
    warn("parser", OWN_CONNECTION),
    warn("parser_class", OWN_CONNECTION),
    read("password").env().in_url(FromUrl::Any),
    warn("path", OWN_CLUSTER),
    warn("policy_resolver", OWN_CLUSTER),
    read("port").env(),
    warn("protocol", NO_EQUIVALENT).env(),
    warn("read_from_replicas", OWN_CLUSTER),
    warn(
        "redis_connect_func",
        "the Python object is not callable from Rust, so static username and password are used",
    )
    .env(),
    warn("reinitialize_steps", OWN_CLUSTER),
    warn("require_full_coverage", OWN_CLUSTER),
    warn("retry", OWN_STACK),
    warn("retry_on_error", NO_EQUIVALENT).env(),
    warn("retry_on_timeout", NO_EQUIVALENT).env(),
    read("sentinel_nodes").or_secrets(&["REDIS_SENTINEL_NODES"]),
    read("sentinel_password").or_secrets(&["REDIS_SENTINEL_PASSWORD"]),
    read("service_name").or_secrets(&["REDIS_SERVICE_NAME"]),
    warn("single_connection_client", NO_EQUIVALENT).env(),
    read("socket_connect_timeout").env().in_url(FromUrl::Any),
    read("socket_keepalive").env(),
    warn("socket_keepalive_options", NO_EQUIVALENT).env(),
    warn("socket_read_size", OWN_CONNECTION),
    read("socket_timeout").env().in_url(FromUrl::Any),
    warn("socket_type", OWN_CONNECTION),
    read("ssl").env(),
    read("ssl_ca_certs").env().in_url(FromUrl::Rediss),
    read("ssl_ca_data").env().in_url(FromUrl::Rediss),
    read("ssl_ca_path").env().in_url(FromUrl::Rediss),
    read("ssl_cert_reqs").env().in_url(FromUrl::Rediss),
    read("ssl_certfile").env().in_url(FromUrl::Rediss),
    read("ssl_check_hostname").env().in_url(FromUrl::Rediss),
    warn("ssl_ciphers", RUSTLS).env(),
    reject("ssl_exclude_verify_flags", VERIFY_FLAGS, Asked::WhenSet),
    reject("ssl_include_verify_flags", VERIFY_FLAGS, Asked::WhenSet),
    read("ssl_keyfile").env().in_url(FromUrl::Rediss),
    warn_unless("ssl_min_version", RUSTLS, is_met_by_rustls).env(),
    warn("ssl_ocsp_context", OCSP).env(),
    warn("ssl_ocsp_expected_cert", OCSP).env(),
    read("ssl_password").env().in_url(FromUrl::Rediss),
    reject("ssl_validate_ocsp", REVOCATION, Asked::WhenTrue).env(),
    reject("ssl_validate_ocsp_stapled", REVOCATION, Asked::WhenTrue).env(),
    read("startup_nodes").or_secrets(&["REDIS_CLUSTER_NODES"]),
    read("unix_socket_path").env(),
    read("url").env(),
    read("username").env().in_url(FromUrl::Any),
];

// rustls only speaks TLS 1.2 and 1.3, so a minimum of 1.2 or lower is already met. redis-py takes an
// ssl.TLSVersion, which reaches Rust as its name or its value
fn is_met_by_rustls(value: &Value) -> bool {
    match value {
        Value::Number(number) => matches!(number.as_i64(), Some(-2 | 769 | 770 | 771)),
        Value::String(text) => matches!(
            text.strip_prefix("TLSVersion.")
                .unwrap_or(text)
                .to_ascii_uppercase()
                .as_str(),
            "MINIMUM_SUPPORTED" | "TLSV1" | "TLSV1_1" | "TLSV1_2"
        ),
        _ => false,
    }
}

pub fn setting(kwarg: &str) -> Option<&'static Setting> {
    SETTINGS.iter().find(|setting| setting.kwarg == kwarg)
}

/// The check a `Reject` row stands for and which values ask for it.
pub(crate) fn refusal(kwarg: &str) -> Option<(&'static str, Asked)> {
    match setting(kwarg)?.handling {
        Handling::Reject { check, asked } => Some((check, asked)),
        Handling::Read | Handling::Warn { .. } => None,
    }
}

/// A warned setting the kwargs set, with what dropping it means. A kwarg with no row is not a Redis setting, so it
/// is dropped without a warning.
pub(crate) fn ignored(source: &str, kwarg: &str, value: &Value) -> Option<Ignored> {
    let consequence = setting(kwarg)?.warning(value)?;
    Some(Ignored {
        setting: source.to_owned(),
        consequence,
    })
}

/// A url query parameter the resolver does not read. redis-py turns every one into a client kwarg, so even one with
/// no row is reported.
pub(crate) fn ignored_query(source: &str, name: &str, value: &Value) -> Option<Ignored> {
    let consequence = match setting(name) {
        None => NO_EQUIVALENT,
        Some(setting) if setting.is_read() => {
            "the Rust client does not read it from this url's query"
        }
        Some(setting) => setting.warning(value)?,
    };
    Some(Ignored {
        setting: source.to_owned(),
        consequence,
    })
}
