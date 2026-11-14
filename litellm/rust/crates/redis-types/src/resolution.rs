use std::time::Duration;

use crate::{Auth, Target};

/// One Redis connection with every setting, environment value and secret already applied. Each
/// client library builds its client from this one value, so they cannot read a setting differently.
#[derive(Clone, Debug, PartialEq)]
pub struct Resolution {
    pub target: Target,
    pub auth: Auth,
    pub namespace: Option<String>,
    pub timeout: Option<Duration>,
    pub connect_timeout: Option<Duration>,
    pub health_check: HealthCheck,
    /// redis-py's `socket_keepalive`, on unless set false, so an idle connection a load balancer
    /// dropped is noticed.
    pub keepalive: bool,
    pub ignored: Vec<Ignored>,
}

/// redis-py's `health_check_interval`, which `_coerce_redis_kwargs_types` reads as an `int` of
/// seconds. Unset and zero differ, because a client may give an unset interval its own default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HealthCheck {
    Unset,
    Disabled,
    Every(Duration),
}

/// A setting the Rust client accepts without honoring, so the host can log what ignoring it means.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ignored {
    /// The key, the environment variable or the url query parameter that set it.
    pub setting: String,
    pub consequence: &'static str,
}
