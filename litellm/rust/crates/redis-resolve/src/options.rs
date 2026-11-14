use std::time::Duration;

use litellm_python_compat::{BigInt, number};
use litellm_redis_types::Ignored;
use litellm_secret_value::SecretValue;
use serde_json::Value;

use crate::{
    Entry, Error, Expected, Kwargs,
    keys::{self, Asked},
};

/// Every setting the resolver reads. A setting is `None` when it is unset, null or empty, except that a null
/// `tls.cert_reqs` is kept because redis-py reads `ssl_cert_reqs=None` as `CERT_NONE`.
pub(super) struct Options {
    pub(super) url: Option<Entry>,
    pub(super) host: Option<Entry>,
    pub(super) port: Option<Entry>,
    pub(super) db: Option<Entry>,
    pub(super) unix_socket_path: Option<Entry>,
    pub(super) startup_nodes: Option<Entry>,
    pub(super) sentinel: SentinelOptions,
    pub(super) username: Option<Entry>,
    pub(super) password: Option<Entry>,
    pub(super) credential_provider: Option<Entry>,
    pub(super) gcp: GcpOptions,
    pub(super) azure: AzureOptions,
    pub(super) aws: AwsOptions,
    pub(super) ssl: Option<Entry>,
    pub(super) tls: TlsInput,
    pub(super) namespace: Option<Entry>,
    pub(super) socket_timeout: Option<Entry>,
    pub(super) socket_connect_timeout: Option<Entry>,
    pub(super) health_check_interval: Option<Entry>,
    pub(super) socket_keepalive: Option<Entry>,
}

pub(super) struct SentinelOptions {
    pub(super) nodes: Option<Entry>,
    pub(super) service_name: Option<Entry>,
    pub(super) password: Option<Entry>,
}

pub(super) struct GcpOptions {
    pub(super) service_account: Option<Entry>,
    pub(super) ssl_ca_certs: Option<Entry>,
}

pub(super) struct AzureOptions {
    pub(super) ad_token: Option<Entry>,
    pub(super) client_id: Option<Entry>,
    pub(super) tenant_id: Option<Entry>,
    pub(super) client_secret: Option<Entry>,
}

pub(super) struct AwsOptions {
    pub(super) auth: Option<Entry>,
    pub(super) user_name: Option<Entry>,
    pub(super) cache_name: Option<Entry>,
    pub(super) region: Option<Entry>,
    pub(super) serverless: Option<Entry>,
}

/// The TLS settings, read from the kwargs or from a `rediss://` url's query.
pub(super) struct TlsInput {
    pub(super) cert_reqs: Option<Entry>,
    pub(super) ca_certs: Option<Entry>,
    pub(super) ca_data: Option<Entry>,
    pub(super) ca_path: Option<Entry>,
    pub(super) certfile: Option<Entry>,
    pub(super) keyfile: Option<Entry>,
    pub(super) key_password: Option<Entry>,
    pub(super) check_hostname: Option<Entry>,
}

/// The settings the kwargs set that the Rust client drops, once none of them asks for a check it cannot perform.
fn ignored(kwargs: &Kwargs) -> Result<Vec<Ignored>, Error> {
    refuse_unenforceable(kwargs.iter())?;
    Ok(kwargs
        .iter()
        .filter_map(|(key, entry)| keys::ignored(&entry.source, key, &entry.value))
        .collect())
}

impl Options {
    /// The typed settings and the ones the Rust client drops.
    pub(super) fn read(kwargs: &Kwargs) -> Result<(Self, Vec<Ignored>), Error> {
        let ignored = ignored(kwargs)?;
        let options = Self::from_lookup(|key| kwargs.get(key).cloned());
        Ok((options, ignored))
    }

    fn from_lookup(lookup: impl Fn(&'static str) -> Option<Entry>) -> Self {
        let get = |key| lookup(key).filter(|entry| is_present(&entry.value));
        Self {
            url: get("url"),
            host: get("host"),
            port: get("port"),
            db: get("db"),
            unix_socket_path: get("unix_socket_path"),
            startup_nodes: get("startup_nodes"),
            sentinel: SentinelOptions {
                nodes: get("sentinel_nodes"),
                service_name: get("service_name"),
                password: get("sentinel_password"),
            },
            username: get("username"),
            password: get("password"),
            credential_provider: get("credential_provider"),
            gcp: GcpOptions {
                service_account: get("gcp_service_account"),
                ssl_ca_certs: get("gcp_ssl_ca_certs"),
            },
            azure: AzureOptions {
                ad_token: get("azure_redis_ad_token"),
                client_id: get("azure_client_id"),
                tenant_id: get("azure_tenant_id"),
                client_secret: get("azure_client_secret"),
            },
            aws: AwsOptions {
                auth: get("aws_iam_auth"),
                user_name: get("aws_iam_user_name"),
                cache_name: get("aws_iam_cache_name"),
                region: get("aws_iam_region"),
                serverless: get("aws_iam_serverless"),
            },
            ssl: get("ssl"),
            tls: TlsInput {
                cert_reqs: lookup("ssl_cert_reqs")
                    .filter(|entry| entry.value.is_null() || is_present(&entry.value)),
                ca_certs: get("ssl_ca_certs"),
                ca_data: get("ssl_ca_data"),
                ca_path: get("ssl_ca_path"),
                certfile: get("ssl_certfile"),
                keyfile: get("ssl_keyfile"),
                key_password: get("ssl_password"),
                check_hostname: get("ssl_check_hostname"),
            },
            namespace: get("namespace"),
            socket_timeout: get("socket_timeout"),
            socket_connect_timeout: get("socket_connect_timeout"),
            health_check_interval: get("health_check_interval"),
            socket_keepalive: get("socket_keepalive"),
        }
    }
}

/// A `Reject` row whose value asks for its check fails, because Rust cannot perform it.
pub(super) fn refuse_unenforceable<'a>(
    entries: impl IntoIterator<Item = (&'a String, &'a Entry)>,
) -> Result<(), Error> {
    entries
        .into_iter()
        .filter_map(|(key, entry)| Some((keys::refusal(key)?, entry)))
        .try_for_each(|((check, asked), entry)| {
            let asks = match asked {
                Asked::WhenTrue => flag(Some(entry))?,
                Asked::WhenSet => is_present(&entry.value),
            };
            match asks {
                true => Err(Error::Unenforceable {
                    setting: entry.source.clone(),
                    check,
                }),
                false => Ok(()),
            }
        })
}

// Python reads most of these with `x or fallback`, so an empty value counts as unset everywhere
pub fn is_present(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(entries) => !entries.is_empty(),
        Value::Bool(_) | Value::Number(_) => true,
    }
}

pub(super) fn invalid(setting: &str, expected: Expected) -> Error {
    Error::Invalid {
        setting: setting.to_owned(),
        expected,
    }
}

pub(super) fn as_text(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(text.clone()),
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(flag) => Some(flag.to_string()),
        _ => None,
    }
}

pub(super) fn text(entry: Option<&Entry>) -> Option<String> {
    entry.and_then(|entry| as_text(&entry.value))
}

pub(super) fn secret(entry: Option<&Entry>) -> Option<SecretValue> {
    text(entry).map(SecretValue::new)
}

/// A flag every Python surface reads the same way. `_str_to_bool` takes only `true`, `1` and `yes`
/// as true while the proxy's pydantic models also take `on`, `t` and `y`, so text they disagree on,
/// or text neither accepts, is an error instead of a guess.
pub(super) fn flag(entry: Option<&Entry>) -> Result<bool, Error> {
    let Some(entry) = entry else {
        return Ok(false);
    };
    match &entry.value {
        Value::Bool(flag) => Ok(*flag),
        Value::Number(number) => Ok(number.as_f64().is_some_and(|number| number != 0.0)),
        Value::String(text) => match text.to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" => Ok(true),
            "false" | "0" | "no" | "off" | "f" | "n" => Ok(false),
            _ => Err(invalid(&entry.source, Expected::Boolean)),
        },
        _ => Err(invalid(&entry.source, Expected::Boolean)),
    }
}

pub(super) fn integer<T: TryFrom<BigInt>>(entry: Option<&Entry>) -> Result<Option<T>, Error> {
    coerced(entry, Expected::Integer, number::int, |number| {
        T::try_from(number).ok()
    })
}

pub(super) fn seconds(entry: Option<&Entry>) -> Result<Option<Duration>, Error> {
    coerced(entry, Expected::Seconds, number::float, |seconds| {
        Duration::try_from_secs_f64(seconds).ok()
    })
}

fn coerced<N, T>(
    entry: Option<&Entry>,
    expected: Expected,
    parse: impl Fn(&str) -> Option<N>,
    convert: impl Fn(N) -> Option<T>,
) -> Result<Option<T>, Error> {
    let Some((entry, text)) = entry.and_then(|entry| Some((entry, as_text(&entry.value)?))) else {
        return Ok(None);
    };
    match parse(&text) {
        Some(number) => convert(number)
            .map(Some)
            .ok_or_else(|| invalid(&entry.source, expected)),
        None => {
            tracing::warn!(setting = %entry.source, "Redis setting is not a number, so the default is used like redis-py");
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, collections::BTreeSet};

    use rstest::rstest;

    use super::Options;
    use crate::SETTINGS;

    // a Read row that Options never asks for would be neither modeled nor passed through nor warned about
    #[rstest]
    fn options_ask_for_exactly_the_read_rows() {
        let asked = RefCell::new(BTreeSet::new());
        let _ = Options::from_lookup(|kwarg| {
            asked.borrow_mut().insert(kwarg);
            None
        });

        let read: BTreeSet<&str> = SETTINGS
            .iter()
            .filter(|setting| setting.is_read())
            .map(|setting| setting.kwarg)
            .collect();
        assert_eq!(asked.into_inner(), read);
    }
}
