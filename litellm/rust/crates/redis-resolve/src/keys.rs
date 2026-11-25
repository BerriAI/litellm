use std::str::FromStr;

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

/// How `_merge_redis_kwargs` reads a fallback: `get_secret` turns `true` and `false` into a bool, and
/// `get_secret_str` treats that bool as unset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Read {
    Secret,
    SecretStr,
}

/// The condition `_merge_redis_kwargs` reads a fallback under, so a secret Python never asks for is never
/// filled in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum When {
    Always,
    /// No `credential_provider` is set.
    WithoutProvider,
    /// Azure AD is on and GCP IAM is not.
    AzureAd,
    /// AWS IAM is on and neither GCP IAM nor Azure AD is.
    AwsIam,
}

/// The names `_merge_redis_kwargs` reads when the merged kwargs lack the setting.
#[derive(Clone, Copy, Debug)]
pub struct Fallback {
    pub names: &'static [&'static str],
    pub read: Read,
    pub when: When,
}

#[derive(Clone, Copy, Debug)]
pub struct Setting {
    pub handling: Handling,
    pub from_url: FromUrl,
    /// `_get_redis_env_kwarg_mapping` reads it as `REDIS_<KWARG>`.
    pub environment: bool,
    pub fallback: Option<Fallback>,
    /// It names where Redis is, so a configured one hides the environment's url.
    pub target: bool,
    /// `Cache` names it as a parameter, so a null `cache_params` value is dropped.
    pub named_by_cache: bool,
}

impl Setting {
    const fn new(handling: Handling) -> Self {
        Self {
            handling,
            from_url: FromUrl::No,
            environment: false,
            fallback: None,
            target: false,
            named_by_cache: false,
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

    const fn or_secrets(self, when: When, read: Read, names: &'static [&'static str]) -> Self {
        Self {
            fallback: Some(Fallback { names, read, when }),
            ..self
        }
    }

    const fn target(self) -> Self {
        Self {
            target: true,
            ..self
        }
    }

    const fn named_by_cache(self) -> Self {
        Self {
            named_by_cache: true,
            ..self
        }
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

const fn read() -> Setting {
    Setting::new(Handling::Read)
}

const fn warn(consequence: &'static str) -> Setting {
    Setting::new(Handling::Warn {
        consequence,
        met: None,
    })
}

const fn warn_unless(consequence: &'static str, met: fn(&Value) -> bool) -> Setting {
    Setting::new(Handling::Warn {
        consequence,
        met: Some(met),
    })
}

const fn reject(check: &'static str, asked: Asked) -> Setting {
    Setting::new(Handling::Reject { check, asked })
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
const NO_CONNECTION_CAP: &str = "the Rust client has no connection cap";
const NOT_CALLABLE: &str =
    "the Python object is not callable from Rust, so static username and password are used";

/// Every Redis kwarg redis-py or `litellm/_redis.py` accepts, spelled in snake case. `redis-py-kwargs.json`
/// records which redis-py versions and `REDIS_*` names accept them, and `tests/catalog.rs` keeps the two in step.
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    Hash,
    strum::EnumIter,
    strum::EnumString,
    strum::IntoStaticStr,
)]
#[strum(serialize_all = "snake_case")]
pub enum Kwarg {
    AddressRemap,
    AutoCloseConnectionPool,
    AwsIamAuth,
    AwsIamCacheName,
    AwsIamRegion,
    AwsIamServerless,
    AwsIamUserName,
    AzureClientId,
    AzureClientSecret,
    AzureRedisAdToken,
    AzureTenantId,
    Cache,
    CacheConfig,
    Charset,
    ClientName,
    ClusterErrorRetryAttempts,
    CommandPacker,
    ConnectionClass,
    ConnectionErrorRetryAttempts,
    ConnectionPool,
    CredentialProvider,
    Db,
    DecodeResponses,
    DriverInfo,
    DynamicStartupNodes,
    EncoderClass,
    Encoding,
    EncodingErrors,
    Errors,
    EventDispatcher,
    GcpServiceAccount,
    GcpSslCaCerts,
    HealthCheckInterval,
    Host,
    LegacyResponses,
    LibName,
    LibVersion,
    LoadBalancingStrategy,
    MaintNotificationsConfig,
    MaintNotificationsPoolHandler,
    MaintenanceNotificationHash,
    MaintenanceState,
    MaxConnections,
    Namespace,
    OrigHostAddress,
    OrigSocketConnectTimeout,
    OrigSocketTimeout,
    OssClusterMaintNotificationsHandler,
    Parser,
    ParserClass,
    Password,
    Path,
    PolicyResolver,
    Port,
    Protocol,
    ReadFromReplicas,
    RedisConnectFunc,
    ReinitializeSteps,
    RequireFullCoverage,
    Retry,
    RetryOnError,
    RetryOnTimeout,
    SentinelNodes,
    SentinelPassword,
    ServiceName,
    SingleConnectionClient,
    SocketConnectTimeout,
    SocketKeepalive,
    SocketKeepaliveOptions,
    SocketReadSize,
    SocketTimeout,
    SocketType,
    Ssl,
    SslCaCerts,
    SslCaData,
    SslCaPath,
    SslCertReqs,
    SslCertfile,
    SslCheckHostname,
    SslCiphers,
    SslExcludeVerifyFlags,
    SslIncludeVerifyFlags,
    SslKeyfile,
    SslMinVersion,
    SslOcspContext,
    SslOcspExpectedCert,
    SslPassword,
    SslValidateOcsp,
    SslValidateOcspStapled,
    StartupNodes,
    UnixSocketPath,
    Url,
    Username,
}

impl Kwarg {
    pub fn name(self) -> &'static str {
        self.into()
    }

    pub fn environment_name(self) -> Option<String> {
        self.setting()
            .environment
            .then(|| format!("REDIS_{}", self.name().to_ascii_uppercase()))
    }

    /// Every environment or secret name the merge can read the kwarg from.
    pub fn source_names(self) -> impl Iterator<Item = String> {
        let fallback = self
            .setting()
            .fallback
            .map_or(&[][..], |fallback| fallback.names);
        self.environment_name()
            .into_iter()
            .chain(fallback.iter().map(|name| (*name).to_owned()))
    }

    /// The whole answer for the kwarg: how it is handled and where else it is read from.
    pub fn setting(self) -> Setting {
        match self {
            Self::AddressRemap => warn(OWN_CLUSTER),
            Self::AutoCloseConnectionPool => warn(OWN_STACK),
            Self::AwsIamAuth => read().env(),
            Self::AwsIamCacheName => read().env(),
            Self::AwsIamRegion => read().env().or_secrets(
                When::AwsIam,
                Read::SecretStr,
                &["AWS_REGION", "AWS_DEFAULT_REGION"],
            ),
            Self::AwsIamServerless => read().env(),
            Self::AwsIamUserName => read().env(),
            Self::AzureClientId => {
                read()
                    .env()
                    .or_secrets(When::AzureAd, Read::SecretStr, &["AZURE_CLIENT_ID"])
            }
            Self::AzureClientSecret => {
                read()
                    .env()
                    .or_secrets(When::AzureAd, Read::SecretStr, &["AZURE_CLIENT_SECRET"])
            }
            Self::AzureRedisAdToken => read().env().or_secrets(
                When::WithoutProvider,
                Read::Secret,
                &["REDIS_AZURE_AD_TOKEN"],
            ),
            Self::AzureTenantId => {
                read()
                    .env()
                    .or_secrets(When::AzureAd, Read::SecretStr, &["AZURE_TENANT_ID"])
            }
            Self::Cache => warn(NO_EQUIVALENT).env(),
            Self::CacheConfig => warn(NO_EQUIVALENT).env(),
            Self::Charset => warn(NO_EQUIVALENT).env(),
            Self::ClientName => warn(NO_EQUIVALENT).env(),
            Self::ClusterErrorRetryAttempts => warn(OWN_CLUSTER),
            Self::CommandPacker => warn(OWN_CONNECTION),
            Self::ConnectionClass => warn(OWN_STACK),
            Self::ConnectionErrorRetryAttempts => warn(OWN_CLUSTER),
            Self::ConnectionPool => warn(OWN_STACK),
            Self::CredentialProvider => read(),
            Self::Db => read().env().in_url(FromUrl::Any),
            Self::DecodeResponses => warn(NO_EQUIVALENT).env(),
            Self::DriverInfo => warn(NO_EQUIVALENT),
            Self::DynamicStartupNodes => warn(OWN_CLUSTER),
            Self::EncoderClass => warn(OWN_CONNECTION),
            Self::Encoding => warn(NO_EQUIVALENT).env(),
            Self::EncodingErrors => warn(NO_EQUIVALENT).env(),
            Self::Errors => warn(NO_EQUIVALENT).env(),
            Self::EventDispatcher => warn(OWN_STACK).env(),
            Self::GcpServiceAccount => read().env().named_by_cache().or_secrets(
                When::WithoutProvider,
                Read::SecretStr,
                &["REDIS_GCP_SERVICE_ACCOUNT"],
            ),
            Self::GcpSslCaCerts => read().env().named_by_cache().or_secrets(
                When::WithoutProvider,
                Read::SecretStr,
                &["REDIS_GCP_SSL_CA_CERTS"],
            ),
            Self::HealthCheckInterval => read().env().in_url(FromUrl::Any),
            Self::Host => read().env().target().named_by_cache(),
            Self::LegacyResponses => warn(NO_EQUIVALENT),
            Self::LibName => warn(NO_EQUIVALENT).env(),
            Self::LibVersion => warn(NO_EQUIVALENT).env(),
            Self::LoadBalancingStrategy => warn(OWN_CLUSTER),
            Self::MaintNotificationsConfig => warn(MAINTENANCE),
            Self::MaintNotificationsPoolHandler => warn(MAINTENANCE),
            Self::MaintenanceNotificationHash => warn(MAINTENANCE),
            Self::MaintenanceState => warn(MAINTENANCE),
            Self::MaxConnections => warn(NO_CONNECTION_CAP).env(),
            Self::Namespace => read().named_by_cache(),
            Self::OrigHostAddress => warn(MAINTENANCE),
            Self::OrigSocketConnectTimeout => warn(MAINTENANCE),
            Self::OrigSocketTimeout => warn(MAINTENANCE),
            Self::OssClusterMaintNotificationsHandler => warn(MAINTENANCE),
            Self::Parser => warn(OWN_CONNECTION),
            Self::ParserClass => warn(OWN_CONNECTION),
            Self::Password => read().env().in_url(FromUrl::Any).named_by_cache(),
            Self::Path => warn(OWN_CLUSTER),
            Self::PolicyResolver => warn(OWN_CLUSTER),
            Self::Port => read().env().named_by_cache(),
            Self::Protocol => warn(NO_EQUIVALENT).env(),
            Self::ReadFromReplicas => warn(OWN_CLUSTER),
            Self::RedisConnectFunc => warn(NOT_CALLABLE).env(),
            Self::ReinitializeSteps => warn(OWN_CLUSTER),
            Self::RequireFullCoverage => warn(OWN_CLUSTER),
            Self::Retry => warn(OWN_STACK),
            Self::RetryOnError => warn(NO_EQUIVALENT).env(),
            Self::RetryOnTimeout => warn(NO_EQUIVALENT).env(),
            Self::SentinelNodes => {
                read()
                    .target()
                    .or_secrets(When::Always, Read::Secret, &["REDIS_SENTINEL_NODES"])
            }
            Self::SentinelPassword => {
                read().or_secrets(When::Always, Read::SecretStr, &["REDIS_SENTINEL_PASSWORD"])
            }
            Self::ServiceName => {
                read().or_secrets(When::Always, Read::Secret, &["REDIS_SERVICE_NAME"])
            }
            Self::SingleConnectionClient => warn(NO_EQUIVALENT).env(),
            Self::SocketConnectTimeout => read().env().in_url(FromUrl::Any),
            Self::SocketKeepalive => read().env(),
            Self::SocketKeepaliveOptions => warn(NO_EQUIVALENT).env(),
            Self::SocketReadSize => warn(OWN_CONNECTION),
            Self::SocketTimeout => read().env().in_url(FromUrl::Any).named_by_cache(),
            Self::SocketType => warn(OWN_CONNECTION),
            Self::Ssl => read().env(),
            Self::SslCaCerts => read().env().in_url(FromUrl::Rediss),
            Self::SslCaData => read().env().in_url(FromUrl::Rediss),
            Self::SslCaPath => read().env().in_url(FromUrl::Rediss),
            Self::SslCertReqs => read().env().in_url(FromUrl::Rediss),
            Self::SslCertfile => read().env().in_url(FromUrl::Rediss),
            Self::SslCheckHostname => read().env().in_url(FromUrl::Rediss),
            Self::SslCiphers => warn(RUSTLS).env(),
            Self::SslExcludeVerifyFlags => reject(VERIFY_FLAGS, Asked::WhenSet),
            Self::SslIncludeVerifyFlags => reject(VERIFY_FLAGS, Asked::WhenSet),
            Self::SslKeyfile => read().env().in_url(FromUrl::Rediss),
            Self::SslMinVersion => warn_unless(RUSTLS, is_met_by_rustls).env(),
            Self::SslOcspContext => warn(OCSP).env(),
            Self::SslOcspExpectedCert => warn(OCSP).env(),
            Self::SslPassword => read().env().in_url(FromUrl::Rediss),
            Self::SslValidateOcsp => reject(REVOCATION, Asked::WhenTrue).env(),
            Self::SslValidateOcspStapled => reject(REVOCATION, Asked::WhenTrue).env(),
            Self::StartupNodes => read().target().named_by_cache().or_secrets(
                When::Always,
                Read::Secret,
                &["REDIS_CLUSTER_NODES"],
            ),
            Self::UnixSocketPath => read().env(),
            Self::Url => read().env().target(),
            Self::Username => read().env().in_url(FromUrl::Any),
        }
    }
}

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

/// A name with no variant is not a Redis setting.
pub(crate) fn setting(kwarg: &str) -> Option<Setting> {
    Kwarg::from_str(kwarg).ok().map(Kwarg::setting)
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
