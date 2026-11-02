#[derive(Debug, thiserror::Error)]
pub enum RedisError {
    #[error("Redis host does not yet support configured setting {setting}")]
    Unsupported { setting: String },
    #[error("invalid Redis setting {setting}: expected {expected}")]
    Invalid { setting: String, expected: Expected },
    #[error("either host or url must be specified for redis")]
    NoTarget,
    #[error("service_name is required with sentinel_nodes")]
    NoSentinelService,
    #[error("{setting} cannot be combined with sentinel_nodes, which redis-py rejects at startup")]
    SentinelWithHost { setting: String },
    #[error("{setting} must be 0 in cluster mode")]
    DbInCluster { setting: String },
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
    #[strum(serialize = "none, optional or required")]
    CertRequirement,
    #[strum(serialize = "a list of objects with host and port")]
    StartupNodes,
    #[strum(serialize = "a non-empty list of [host, port] pairs")]
    SentinelNodes,
}
