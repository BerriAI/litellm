use litellm_config::Value;
use litellm_python_compat::{
    flags::{self, lower_is_true},
    json::from_json,
};
use strum::IntoEnumIterator;

use crate::{
    Entry, Error, Kwargs,
    configured::{ConfiguredRedis, Nulls},
    is_present,
    keys::{self, Fallback, Kwarg, Read, When},
    secrets::SecretSnapshot,
};

/// The configured settings over their `REDIS_*` values, `{**environment_kwargs, **env_overrides}`
/// in `_merge_redis_kwargs`, with the `get_secret` fallbacks it reads filled in. A null left in the result is
/// an explicit `None` that hid its environment value.
pub(crate) fn merged(
    configured: &ConfiguredRedis,
    secrets: &SecretSnapshot,
) -> Result<Kwargs, Error> {
    let block: Kwargs = configured
        .fields
        .iter()
        .filter_map(|(key, value)| match configured_value(key, value, secrets) {
            Ok(value) if value.is_null() && !keeps_null(configured.nulls, key) => None,
            Ok(value) => {
                let entry = Entry {
                    value,
                    source: key.clone(),
                };
                Some(Ok((key.clone(), entry)))
            }
            Err(error) => Some(Err(error)),
        })
        .collect::<Result<_, Error>>()?;
    let url = Kwarg::Url.name();
    let names_a_target = Kwarg::iter()
        .filter(|kwarg| kwarg.setting().target && *kwarg != Kwarg::Url)
        .any(|kwarg| is_set(&block, kwarg.name()));
    let drop_environment_url = names_a_target && !is_set(&block, url);
    let layered: Kwargs = environment(secrets)
        .filter(|(key, _)| !(key == url && drop_environment_url))
        .chain(block)
        .collect();
    let fallbacks = fallbacks(&layered, configured.credential_provider.is_some(), secrets);
    Ok(layered.into_iter().chain(fallbacks).collect())
}

fn fallbacks(layered: &Kwargs, has_provider: bool, secrets: &SecretSnapshot) -> Kwargs {
    let fill = |when: When| -> Kwargs {
        Kwarg::iter()
            .filter_map(|kwarg| {
                let fallback = kwarg.setting().fallback?;
                (fallback.when == when).then_some((kwarg, fallback))
            })
            .filter_map(|(kwarg, fallback)| filled(layered, kwarg, fallback, secrets))
            .collect()
    };
    let always = fill(When::Always);
    if has_provider || is_set(layered, Kwarg::CredentialProvider.name()) {
        return always;
    }
    let without_provider = fill(When::WithoutProvider);
    let value = |kwarg: Kwarg| {
        layered
            .get(kwarg.name())
            .filter(|entry| is_present(&entry.value))
            .or_else(|| without_provider.get(kwarg.name()))
            .map(|entry| &entry.value)
    };
    let gcp = value(Kwarg::GcpServiceAccount).is_some();
    let azure = value(Kwarg::AzureRedisAdToken)
        .is_some_and(|token| text(token).is_some_and(|token| lower_is_true(&token)));
    let aws = layered
        .get(Kwarg::AwsIamAuth.name())
        .is_some_and(|entry| flags::coerces_to_true(&from_json(entry.value.clone())));
    let azure_settings = match azure && !gcp {
        true => fill(When::AzureAd),
        false => Kwargs::new(),
    };
    let aws_settings = match aws && !gcp && !azure {
        true => fill(When::AwsIam),
        false => Kwargs::new(),
    };
    always
        .into_iter()
        .chain(without_provider)
        .chain(azure_settings)
        .chain(aws_settings)
        .collect()
}

fn filled(
    layered: &Kwargs,
    kwarg: Kwarg,
    fallback: Fallback,
    secrets: &SecretSnapshot,
) -> Option<(String, Entry)> {
    if layered
        .get(kwarg.name())
        .is_some_and(|entry| is_present(&entry.value))
    {
        return None;
    }
    fallback.names.iter().find_map(|name| {
        let value = secrets
            .get(name)
            .filter(|value| value.as_str() != Some(""))?;
        match (fallback.read, &value) {
            (Read::SecretStr, serde_json::Value::Bool(_)) => None,
            _ => {
                let entry = Entry {
                    value,
                    source: (*name).to_owned(),
                };
                Some((kwarg.name().to_owned(), entry))
            }
        }
    })
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
        Nulls::KeptUnlessNamedByCache => {
            !keys::setting(key).is_some_and(|setting| setting.named_by_cache)
        }
        #[cfg(test)]
        Nulls::Kept => true,
    }
}

fn environment(secrets: &SecretSnapshot) -> impl Iterator<Item = (String, Entry)> + '_ {
    Kwarg::iter().filter_map(|kwarg| {
        let source = kwarg.environment_name()?;
        let value = secrets.get(&source)?;
        let entry = Entry { value, source };
        Some((kwarg.name().to_owned(), entry))
    })
}

/// Whether the secrets alone name where Redis is, under any name the merge reads a target from.
pub(crate) fn secrets_name_a_target(secrets: &SecretSnapshot) -> bool {
    Kwarg::iter()
        .filter(|kwarg| kwarg.setting().target)
        .flat_map(Kwarg::source_names)
        .any(|name| secrets.get(&name).is_some())
}

// load_config resolves os.environ/ to None when unset, so a missing reference is a null like any other
fn configured_value(
    key: &str,
    value: &Value,
    secrets: &SecretSnapshot,
) -> Result<serde_json::Value, Error> {
    let reference = match value {
        Value::String(text) => text.strip_prefix("os.environ/"),
        _ => None,
    };
    match reference {
        Some(name) => Ok(secrets.get(name).unwrap_or(serde_json::Value::Null)),
        None => serde_json::to_value(value).map_err(|_| Error::NotPlainData {
            setting: key.to_owned(),
        }),
    }
}

fn is_set(kwargs: &Kwargs, key: &str) -> bool {
    kwargs.get(key).is_some_and(|entry| !entry.value.is_null())
}
