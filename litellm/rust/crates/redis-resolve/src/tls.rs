use std::time::Duration;

use litellm_python_compat::number;
use litellm_redis_types::{Ignored, TlsOptions};
use litellm_secret_value::SecretValue;

use serde_json::Value;

use crate::{
    Entry, Error, Expected,
    keys::{self, FromUrl},
    options::{self, TlsInput, flag, invalid, secret, text},
};

/// `gcp_ca_certs` replaces `ssl_ca_certs` when GCP IAM is on, as `_get_redis_client_logic` does.
pub(super) fn tls_options(
    input: &TlsInput,
    gcp_ca_certs: Option<&Entry>,
) -> Result<TlsOptions, Error> {
    Ok(TlsOptions {
        ca_certs: text(gcp_ca_certs.or(input.ca_certs.as_ref())).map(Into::into),
        ca_data: text(input.ca_data.as_ref()),
        ca_path: text(input.ca_path.as_ref()).map(Into::into),
        certfile: text(input.certfile.as_ref()).map(Into::into),
        keyfile: text(input.keyfile.as_ref()).map(Into::into),
        key_password: secret(input.key_password.as_ref()),
        verify_certificate: verify_certificate(input.cert_reqs.as_ref())?,
        check_hostname: flag(input.check_hostname.as_ref())?,
    })
}

// RedisSSLContext looks text up case-sensitively and takes an ssl.VerifyMode as is
fn verify_certificate(cert_reqs: Option<&Entry>) -> Result<bool, Error> {
    let Some(entry) = cert_reqs else {
        return Ok(true);
    };
    match &entry.value {
        // Explicit None intentionally means CERT_NONE, matching redis-py rather than the unset default.
        Value::Null => Ok(false),
        Value::String(text) if text == "none" => Ok(false),
        Value::String(text) if text == "optional" || text == "required" => Ok(true),
        Value::Number(mode) => match mode.as_i64() {
            Some(0) => Ok(false),
            Some(1 | 2) => Ok(true),
            _ => Err(invalid(&entry.source, Expected::CertRequirement)),
        },
        _ => Err(invalid(&entry.source, Expected::CertRequirement)),
    }
}

/// What redis-py's `parse_url` reads from a url's query on top of the url itself.
pub(super) struct UrlQuery {
    pub(super) tls: Option<TlsOptions>,
    pub(super) db: Option<i64>,
    pub(super) username: Option<SecretValue>,
    pub(super) password: Option<SecretValue>,
    pub(super) settings: QuerySettings,
}

/// The connection settings a url's query sets, which win over the kwargs because `from_url` applies them last.
#[derive(Default)]
pub(super) struct QuerySettings {
    pub(super) timeout: Option<Duration>,
    pub(super) connect_timeout: Option<Duration>,
    pub(super) health_check_interval: Option<i64>,
    pub(super) ignored: Vec<Ignored>,
}

pub(super) fn url_query(url: &str) -> Result<UrlQuery, Error> {
    let Ok(url) = url::Url::parse(url) else {
        return Ok(UrlQuery {
            tls: None,
            db: None,
            username: None,
            password: None,
            settings: QuerySettings::default(),
        });
    };
    let tls_scheme = url.scheme().eq_ignore_ascii_case("rediss");
    let pairs: Vec<(String, Entry)> = url
        .query_pairs()
        .filter(|(_, value)| !value.is_empty())
        .map(|(name, value)| {
            (
                name.to_string(),
                Entry {
                    value: Value::String(value.into_owned()),
                    source: format!("url ?{name}"),
                },
            )
        })
        .collect();
    // parse_qs keeps every value of a repeated parameter and parse_url reads the first
    let query: Vec<(String, Entry)> = pairs
        .iter()
        .enumerate()
        .filter(|(index, (name, _))| !pairs[..*index].iter().any(|(earlier, _)| earlier == name))
        .map(|(_, pair)| pair.clone())
        .collect();
    options::refuse_unenforceable(query.iter().map(|(name, entry)| (name, entry)))?;
    let (query, ignored): (Vec<_>, Vec<_>) = query.into_iter().partition(|(name, _)| {
        keys::setting(name).is_some_and(|setting| match setting.from_url {
            FromUrl::Any => true,
            FromUrl::Rediss => tls_scheme,
            FromUrl::No => false,
        })
    });
    let get = |key: &str| {
        query
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, entry)| entry.clone())
    };
    let seconds = |key: &str| {
        query_number(
            get(key).as_ref(),
            number::float,
            Expected::Seconds,
            |seconds| Duration::try_from_secs_f64(seconds).ok(),
        )
    };
    let integer = |key: &str| {
        query_number(
            get(key).as_ref(),
            number::int,
            Expected::Integer,
            |number| i64::try_from(number).ok(),
        )
    };
    let db = integer("db")?;
    let settings = QuerySettings {
        timeout: seconds("socket_timeout")?,
        connect_timeout: seconds("socket_connect_timeout")?,
        health_check_interval: integer("health_check_interval")?,
        ignored: ignored
            .iter()
            .filter_map(|(name, entry)| keys::ignored_query(&entry.source, name, &entry.value))
            .collect(),
    };
    let tls = match tls_scheme {
        true => Some(TlsOptions {
            ca_certs: text(get("ssl_ca_certs").as_ref()).map(Into::into),
            ca_data: text(get("ssl_ca_data").as_ref()),
            ca_path: text(get("ssl_ca_path").as_ref()).map(Into::into),
            certfile: text(get("ssl_certfile").as_ref()).map(Into::into),
            keyfile: text(get("ssl_keyfile").as_ref()).map(Into::into),
            key_password: secret(get("ssl_password").as_ref()),
            verify_certificate: verify_certificate(get("ssl_cert_reqs").as_ref())?,
            check_hostname: text(get("ssl_check_hostname").as_ref())
                .is_some_and(|text| url_bool(&text)),
        }),
        false => None,
    };
    Ok(UrlQuery {
        tls,
        db,
        username: secret(get("username").as_ref()),
        password: secret(get("password").as_ref()),
        settings,
    })
}

// parse_url reads these with int() or float() and raises ValueError building the client on anything else
fn query_number<N, T>(
    entry: Option<&Entry>,
    parse: impl Fn(&str) -> Option<N>,
    expected: Expected,
    convert: impl Fn(N) -> Option<T>,
) -> Result<Option<T>, Error> {
    let Some(entry) = entry else {
        return Ok(None);
    };
    let number = text(Some(entry))
        .and_then(|text| parse(&text))
        .ok_or_else(|| Error::UrlQuery {
            setting: entry.source.clone(),
        })?;
    convert(number)
        .map(Some)
        .ok_or_else(|| invalid(&entry.source, expected))
}

// redis-py's to_bool: only 0, F, FALSE, N and NO (any case) are false
fn url_bool(text: &str) -> bool {
    !matches!(
        text.to_ascii_uppercase().as_str(),
        "" | "0" | "F" | "FALSE" | "N" | "NO"
    )
}
