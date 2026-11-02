use litellm_redis_types::TlsOptions;

use super::{Entry, Expected, Kwargs, RedisError, bool_value, parsed, text_value};
use crate::Value;

// redis-py's parse_url turns every query parameter into a client kwarg. Only the TLS ones have a redis-rs
// equivalent, so any other parameter, and any parameter on a plain redis:// url, is rejected
const URL_TLS_SETTINGS: &[&str] = &[
    "ssl_cert_reqs",
    "ssl_ca_certs",
    "ssl_certfile",
    "ssl_keyfile",
    "ssl_check_hostname",
];

pub(super) fn tls_options(
    merged: &Kwargs,
    gcp_ca_certs: Option<String>,
) -> Result<TlsOptions, RedisError> {
    let verify_certificate =
        parsed(
            merged,
            "ssl_cert_reqs",
            Expected::CertRequirement,
            |text| match text {
                "required" | "optional" => Some(true),
                "none" => Some(false),
                _ => None,
            },
        )?
        .unwrap_or(true);
    Ok(TlsOptions {
        ca_certs: gcp_ca_certs
            .or_else(|| text_value(merged, "ssl_ca_certs"))
            .map(Into::into),
        certfile: text_value(merged, "ssl_certfile").map(Into::into),
        keyfile: text_value(merged, "ssl_keyfile").map(Into::into),
        verify_certificate,
        check_hostname: bool_value(merged, "ssl_check_hostname"),
    })
}

pub(super) fn url_tls(url: &str) -> Result<TlsOptions, RedisError> {
    let Ok(parsed) = url::Url::parse(url) else {
        return Ok(TlsOptions::verified());
    };
    let tls_scheme = parsed.scheme().eq_ignore_ascii_case("rediss");
    let query: Kwargs = parsed
        .query_pairs()
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
    if let Some((_, entry)) = query
        .iter()
        .find(|(name, _)| !tls_scheme || !URL_TLS_SETTINGS.contains(&name.as_str()))
    {
        return Err(RedisError::Unsupported {
            setting: entry.source.clone(),
        });
    }
    Ok(TlsOptions {
        check_hostname: text_value(&query, "ssl_check_hostname")
            .is_some_and(|text| url_bool(&text)),
        ..tls_options(&query, None)?
    })
}

// redis-py's to_bool: only 0, F, FALSE, N and NO (any case) are false
fn url_bool(text: &str) -> bool {
    !matches!(
        text.to_ascii_uppercase().as_str(),
        "" | "0" | "F" | "FALSE" | "N" | "NO"
    )
}
