use litellm_python_compat::number;
use litellm_redis_types::{Ignored, TlsOptions};
use litellm_secret_value::SecretValue;

use serde_json::Value;

use crate::{
    Entry, Error, Expected,
    keys::{self, FromUrl},
    options::{self, TlsInput, flag, parsed, secret, text},
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

fn verify_certificate(cert_reqs: Option<&Entry>) -> Result<bool, Error> {
    parsed(cert_reqs, Expected::CertRequirement, |text| match text {
        "required" | "optional" => Some(true),
        "none" => Some(false),
        _ => None,
    })
    .map(|verify| verify.unwrap_or(true))
}

/// What redis-py's `parse_url` reads from a url's query on top of the url itself.
pub(super) struct UrlQuery {
    pub(super) tls: Option<TlsOptions>,
    pub(super) db: Option<i64>,
    pub(super) username: Option<SecretValue>,
    pub(super) password: Option<SecretValue>,
    pub(super) ignored: Vec<Ignored>,
}

pub(super) fn url_query(url: &str) -> Result<UrlQuery, Error> {
    let Ok(url) = url::Url::parse(url) else {
        return Ok(UrlQuery {
            tls: None,
            db: None,
            username: None,
            password: None,
            ignored: Vec::new(),
        });
    };
    let tls_scheme = url.scheme().eq_ignore_ascii_case("rediss");
    let query: Vec<(String, Entry)> = url
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
            .rev()
            .find(|(name, _)| name == key)
            .map(|(_, entry)| entry.clone())
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
        // parse_url reads ?db= with int() and raises on anything else
        db: parsed(get("db").as_ref(), Expected::Integer, |text| {
            number::int(text).and_then(|db| i64::try_from(db).ok())
        })?,
        username: secret(get("username").as_ref()),
        password: secret(get("password").as_ref()),
        ignored: ignored
            .iter()
            .filter_map(|(name, entry)| keys::ignored_query(&entry.source, name, &entry.value))
            .collect(),
    })
}

// redis-py's to_bool: only 0, F, FALSE, N and NO (any case) are false
fn url_bool(text: &str) -> bool {
    !matches!(
        text.to_ascii_uppercase().as_str(),
        "" | "0" | "F" | "FALSE" | "N" | "NO"
    )
}
