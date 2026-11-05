use litellm_config::Value;
use litellm_core_utils::settings::Lookup;
use litellm_redis_resolve::{Entry, Kwargs, SETTINGS, is_present, setting};

use super::configured::{ConfiguredRedis, Nulls};
use crate::Error;

const NAMED_BY_CACHE: &[&str] = &[
    "host",
    "port",
    "password",
    "socket_timeout",
    "startup_nodes",
    "namespace",
    "gcp_service_account",
    "gcp_ssl_ca_certs",
];

/// How `_merge_redis_kwargs` reads a fallback: `get_secret` turns `true` and `false` into a bool, and
/// `get_secret_str` treats that bool as unset.
#[derive(Clone, Copy)]
enum Read {
    Secret,
    SecretStr,
}

const TARGETS: &[(&str, Read)] = &[
    ("startup_nodes", Read::Secret),
    ("sentinel_nodes", Read::Secret),
    ("sentinel_password", Read::SecretStr),
    ("service_name", Read::Secret),
];

const IDENTITIES: &[(&str, Read)] = &[
    ("gcp_service_account", Read::SecretStr),
    ("gcp_ssl_ca_certs", Read::SecretStr),
    ("azure_redis_ad_token", Read::Secret),
];

const AZURE: &[(&str, Read)] = &[
    ("azure_client_id", Read::SecretStr),
    ("azure_tenant_id", Read::SecretStr),
    ("azure_client_secret", Read::SecretStr),
];

const AWS: &[(&str, Read)] = &[("aws_iam_region", Read::SecretStr)];

/// The configured settings over their `REDIS_*` values, `{**environment_kwargs, **env_overrides}`
/// in `_merge_redis_kwargs`, with the `get_secret` fallbacks it reads filled in. A null left in the result is
/// an explicit `None` that hid its environment value.
pub(super) fn merged(configured: &ConfiguredRedis, lookup: &dyn Lookup) -> Result<Kwargs, Error> {
    let block: Kwargs = configured
        .fields
        .iter()
        .map(|(key, value)| (key, environment_reference(value, lookup)))
        .filter(|(key, value)| !value.is_null() || keeps_null(configured.nulls, key))
        .map(|(key, value)| {
            let value = serde_json::to_value(&value).map_err(|_| Error::RedisNotPlainData {
                setting: key.clone(),
            })?;
            let entry = Entry {
                value,
                source: key.clone(),
            };
            Ok((key.clone(), entry))
        })
        .collect::<Result<_, Error>>()?;
    let names_a_target = ["host", "startup_nodes", "sentinel_nodes"]
        .iter()
        .any(|key| is_set(&block, key));
    let drop_environment_url = names_a_target && !is_set(&block, "url");
    let layered: Kwargs = environment(lookup)
        .filter(|(key, _)| !(key == "url" && drop_environment_url))
        .chain(block)
        .collect();
    let fallbacks = fallbacks(&layered, configured.credential_provider.is_some(), lookup);
    Ok(layered.into_iter().chain(fallbacks).collect())
}

// The fallbacks are read under the same conditions as _merge_redis_kwargs, so a secret Python never asks
// for is never filled in
fn fallbacks(layered: &Kwargs, has_provider: bool, lookup: &dyn Lookup) -> Kwargs {
    let fill = |kwargs: &[(&str, Read)]| -> Kwargs {
        kwargs
            .iter()
            .filter_map(|(kwarg, read)| fallback(layered, kwarg, *read, lookup))
            .collect()
    };
    let targets = fill(TARGETS);
    if has_provider || is_set(layered, "credential_provider") {
        return targets;
    }
    let identities = fill(IDENTITIES);
    let value = |kwarg: &str| {
        layered
            .get(kwarg)
            .filter(|entry| is_present(&entry.value))
            .or_else(|| identities.get(kwarg))
            .map(|entry| &entry.value)
    };
    let gcp = value("gcp_service_account").is_some();
    let azure = value("azure_redis_ad_token")
        .is_some_and(|token| text(token).is_some_and(|token| token.eq_ignore_ascii_case("true")));
    let aws = layered
        .get("aws_iam_auth")
        .is_some_and(|entry| coerces_to_true(&entry.value));
    let azure_settings = match azure && !gcp {
        true => fill(AZURE),
        false => Kwargs::new(),
    };
    let aws_settings = match aws && !gcp && !azure {
        true => fill(AWS),
        false => Kwargs::new(),
    };
    targets
        .into_iter()
        .chain(identities)
        .chain(azure_settings)
        .chain(aws_settings)
        .collect()
}

fn fallback(
    layered: &Kwargs,
    kwarg: &str,
    read: Read,
    lookup: &dyn Lookup,
) -> Option<(String, Entry)> {
    if layered
        .get(kwarg)
        .is_some_and(|entry| is_present(&entry.value))
    {
        return None;
    }
    let names = setting(kwarg).map_or(&[][..], |setting| setting.fallback);
    names.iter().find_map(|name| {
        let value = lookup
            .get(name)
            .filter(|value| !value.is_empty())
            .map(secret)?;
        match (read, &value) {
            (Read::SecretStr, serde_json::Value::Bool(_)) => None,
            _ => {
                let entry = Entry {
                    value,
                    source: (*name).to_owned(),
                };
                Some((kwarg.to_owned(), entry))
            }
        }
    })
}

// get_secret reads an environment value of true or false, in any case and padding, as a bool
fn secret_flag(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn secret(value: String) -> serde_json::Value {
    secret_flag(&value).map_or(serde_json::Value::String(value), serde_json::Value::Bool)
}

// _coerces_to_true: text is read like _str_to_bool, anything else by its truthiness
fn coerces_to_true(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::String(text) => {
            matches!(text.to_ascii_lowercase().as_str(), "true" | "1" | "yes")
        }
        serde_json::Value::Null => false,
        serde_json::Value::Bool(flag) => *flag,
        serde_json::Value::Number(number) => number.as_f64().is_some_and(|number| number != 0.0),
        serde_json::Value::Array(items) => !items.is_empty(),
        serde_json::Value::Object(entries) => !entries.is_empty(),
    }
}

fn text(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(text) => Some(text.clone()),
        serde_json::Value::Number(number) => Some(number.to_string()),
        serde_json::Value::Bool(flag) => Some(flag.to_string()),
        _ => None,
    }
}

fn keeps_null(nulls: Nulls, key: &str) -> bool {
    match nulls {
        Nulls::Dropped => false,
        Nulls::KeptUnlessNamedByCache => !NAMED_BY_CACHE.contains(&key),
        #[cfg(test)]
        Nulls::Kept => true,
    }
}

fn environment(lookup: &dyn Lookup) -> impl Iterator<Item = (String, Entry)> + '_ {
    SETTINGS.iter().filter_map(|setting| {
        let source = setting.environment_name()?;
        let value = lookup.get(&source)?;
        let entry = Entry {
            value: secret(value),
            source,
        };
        Some((setting.kwarg.to_owned(), entry))
    })
}

// load_config resolves os.environ/ to None when unset, so a missing reference is a null like any other
fn environment_reference(value: &Value, lookup: &dyn Lookup) -> Value {
    match value {
        Value::String(text) => match text.strip_prefix("os.environ/") {
            Some(name) => lookup.get(name).map_or(Value::Null, |value| {
                secret_flag(&value).map_or(Value::String(value), Value::Bool)
            }),
            None => value.clone(),
        },
        _ => value.clone(),
    }
}

fn is_set(kwargs: &Kwargs, key: &str) -> bool {
    kwargs.get(key).is_some_and(|entry| !entry.value.is_null())
}
