use std::{collections::BTreeMap, time::Duration};

use litellm_redis_types::{Auth, HealthCheck, ProviderHandle, Resolution, StaticAuth, Target};
use serde_json::Value;

mod auth;
mod endpoint;
mod error;
mod keys;
mod options;
mod tls;

pub use error::{Error, Expected, PythonException};
pub use keys::{Asked, FromUrl, Handling, SETTINGS, Setting, setting};
pub use options::is_present;

use options::Options;

/// One merged redis-py kwarg and the name a warning or error reports it under.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub value: Value,
    pub source: String,
}

pub type Kwargs = BTreeMap<String, Entry>;

/// Kwargs whose every entry is reported under its own kwarg name.
pub fn kwargs(fields: impl IntoIterator<Item = (String, Value)>) -> Kwargs {
    fields
        .into_iter()
        .map(|(kwarg, value)| {
            let entry = Entry {
                value,
                source: kwarg.clone(),
            };
            (kwarg, entry)
        })
        .collect()
}

/// What redis-py's constructors make of kwargs a host already merged the way `_get_redis_client_logic` does. A
/// host passes its `credential_provider` object as `provider`, and `REDIS_USERNAME` as its process environment
/// holds it, which Python's Azure AD auth reads from `os.environ` instead of the merged kwargs.
pub fn interpret(
    kwargs: &Kwargs,
    provider: Option<ProviderHandle>,
    redis_username: Option<&str>,
) -> Result<Resolution, Error> {
    let (options, ignored) = Options::read(kwargs)?;
    let auth = auth::redis_auth(&options, provider, redis_username)?;
    let (target, query) = endpoint::target(&options, &auth)?;
    let auth = match (&target, auth) {
        (Target::Url(_), Auth::Static(StaticAuth { username, .. })) => Auth::Static(StaticAuth {
            username,
            password: None,
        }),
        (_, auth) => auth,
    };
    let timeout = query.timeout.map_or_else(
        || options::seconds(options.socket_timeout.as_ref()),
        |timeout| Ok(Some(timeout)),
    )?;
    let connect_timeout = query.connect_timeout.map_or_else(
        || options::seconds(options.socket_connect_timeout.as_ref()),
        |timeout| Ok(Some(timeout)),
    )?;
    let health_check_interval = query.health_check_interval.map_or_else(
        || options::integer::<i64>(options.health_check_interval.as_ref()),
        |interval| Ok(Some(interval)),
    )?;
    Ok(Resolution {
        target,
        auth,
        namespace: options::text(options.namespace.as_ref()),
        timeout,
        connect_timeout,
        health_check: health_check(health_check_interval),
        keepalive: options
            .socket_keepalive
            .as_ref()
            .map_or(Ok(true), |entry| options::flag(Some(entry)))?,
        ignored: ignored.into_iter().chain(query.ignored).collect(),
    })
}

// redis-py checks health only when the interval is truthy, so zero turns it off and a negative interval checks
// before every command
fn health_check(seconds: Option<i64>) -> HealthCheck {
    match seconds {
        None => HealthCheck::Unset,
        Some(0) => HealthCheck::Disabled,
        Some(seconds) => HealthCheck::Every(Duration::from_secs(seconds.try_into().unwrap_or(0))),
    }
}
