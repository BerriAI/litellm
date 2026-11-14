#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid Redis setting {setting}: expected {expected}")]
    Invalid { setting: String, expected: Expected },
    #[error("invalid Redis setting {setting}: redis-py's parse_url cannot read it as a number")]
    UrlQuery { setting: String },
    #[error("either host or url must be specified for redis")]
    NoTarget,
    #[error("service_name is required with sentinel_nodes")]
    NoSentinelService,
    #[error("{setting} cannot be combined with sentinel_nodes, which redis-py rejects at startup")]
    SentinelWithHost { setting: String },
    #[error("{setting} must be 0 in cluster mode")]
    DbInCluster { setting: String },
    #[error("credential_provider is a host object, and the host passed no handle for it")]
    ProviderWithoutHandle,
    #[error(
        "{setting} asks for {check}, which rustls cannot perform, so Redis is not connected without them"
    )]
    Unenforceable {
        setting: String,
        check: &'static str,
    },
    #[error("AWS ElastiCache IAM Redis authentication requires TLS")]
    AwsIamWithoutTls,
    #[error("AWS ElastiCache IAM Redis authentication requires: {}", missing.join(", "))]
    AwsIamIncomplete { missing: Vec<&'static str> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::Display)]
pub enum Expected {
    #[strum(serialize = "an integer")]
    Integer,
    #[strum(serialize = "a boolean")]
    Boolean,
    #[strum(serialize = "a non-negative number of seconds")]
    Seconds,
    #[strum(serialize = "none, optional, required, null or an ssl.VerifyMode value")]
    CertRequirement,
    #[strum(serialize = "a list of objects with host and port")]
    StartupNodes,
    #[strum(serialize = "a non-empty list of [host, port] pairs")]
    SentinelNodes,
}

/// The exception class redis-py 5.3.1 raises building its client.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PythonException {
    ValueError,
    TypeError,
    RedisClusterException,
}

impl Error {
    /// What `_get_redis_client_logic` and redis-py's async client constructors raise on the same kwargs, or `None`
    /// when Python builds the client and fails only once a command opens a connection, or never fails.
    pub fn raised_building_the_python_client(&self) -> Option<PythonException> {
        match self {
            Self::Invalid { expected, .. } => match expected {
                Expected::StartupNodes | Expected::SentinelNodes => {
                    Some(PythonException::ValueError)
                }
                Expected::Integer
                | Expected::Boolean
                | Expected::Seconds
                | Expected::CertRequirement => None,
            },
            Self::UrlQuery { .. }
            | Self::NoTarget
            | Self::NoSentinelService
            | Self::AwsIamWithoutTls
            | Self::AwsIamIncomplete { .. } => Some(PythonException::ValueError),
            Self::SentinelWithHost { .. } | Self::Unenforceable { .. } => {
                Some(PythonException::TypeError)
            }
            Self::DbInCluster { .. } => Some(PythonException::RedisClusterException),
            Self::ProviderWithoutHandle => None,
        }
    }
}
