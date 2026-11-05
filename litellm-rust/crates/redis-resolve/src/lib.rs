use std::{collections::BTreeMap, time::Duration};

use litellm_redis_types::{Auth, HealthCheck, ProviderHandle, Resolution, StaticAuth, Target};
use serde_json::Value;

mod auth;
mod endpoint;
mod error;
mod keys;
mod options;
mod tls;

pub use error::{Error, Expected};
pub use keys::{Asked, FromUrl, Handling, SETTINGS, Setting, Shape, setting};
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
/// host passes its `credential_provider` object as `provider`.
pub fn interpret(kwargs: &Kwargs, provider: Option<ProviderHandle>) -> Result<Resolution, Error> {
    let (options, unmodeled) = Options::read(kwargs)?;
    let auth = auth::redis_auth(&options, provider)?;
    let (target, url_ignored) = endpoint::target(&options, &auth)?;
    let auth = match (&target, auth) {
        (Target::Url(_), Auth::Static(StaticAuth { username, .. })) => Auth::Static(StaticAuth {
            username,
            password: None,
        }),
        (_, auth) => auth,
    };
    Ok(Resolution {
        target,
        auth,
        namespace: options::text(options.namespace.as_ref()),
        timeout: options::seconds(options.socket_timeout.as_ref())?,
        connect_timeout: options::seconds(options.socket_connect_timeout.as_ref())?,
        health_check: health_check(options.health_check_interval.as_ref())?,
        passthrough: unmodeled.passthrough,
        host_objects: unmodeled.host_objects,
        ignored: unmodeled.ignored.into_iter().chain(url_ignored).collect(),
    })
}

// redis-py checks health only when the interval is truthy, so zero turns it off and a negative interval checks
// before every command
fn health_check(entry: Option<&Entry>) -> Result<HealthCheck, Error> {
    Ok(match options::integer::<i64>(entry)? {
        None => HealthCheck::Unset,
        Some(0) => HealthCheck::Disabled,
        Some(seconds) => HealthCheck::Every(Duration::from_secs(seconds.try_into().unwrap_or(0))),
    })
}
