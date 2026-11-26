use std::time::Duration;

use litellm_cache_response::{CacheOptions, CachePolicy, CacheScope};
use litellm_gateway_auth::Authenticated;
use litellm_gateway_auth_core::{Caller, Rows, StoredKey};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::BodyError;

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Controls {
    #[serde(rename = "no-cache")]
    no_cache: bool,
    #[serde(rename = "no-store")]
    no_store: bool,
    ttl: Option<f64>,
    #[serde(rename = "s-maxage", alias = "s-max-age")]
    max_age: Option<f64>,
}

type Prepared = (Map<String, Value>, CacheOptions);

/// Whose answers a cached answer may be served to. A key is named by the hash its row is stored
/// under, so every credential that stands for one key shares its answers.
fn scope<R: Rows<Key: StoredKey>>(caller: &Caller<R>) -> String {
    let [authority, id] = match caller.hash() {
        None => ["litellm:master-key", "master"],
        Some(hash) => ["litellm:virtual-key", hash.as_str()],
    };
    serde_json::json!([authority, id, id]).to_string()
}

pub(crate) fn prepare<R: Rows<Key: StoredKey>>(
    identity: &Authenticated<R>,
    body: Map<String, Value>,
) -> Result<Prepared, BodyError> {
    let controls: Controls = match body.get("cache").filter(|value| !value.is_null()) {
        Some(value) => serde_json::from_value(value.clone()).map_err(BodyError::CacheControls)?,
        None => Controls::default(),
    };
    let caching: Option<bool> = body
        .get("caching")
        .filter(|value| !value.is_null())
        .map(|value| serde_json::from_value(value.clone()))
        .transpose()
        .map_err(BodyError::Caching)?;
    let options = CacheOptions {
        policy: CachePolicy {
            caching,
            no_cache: controls.no_cache,
            no_store: controls.no_store,
            ttl: controls.ttl.map(duration).transpose()?,
            max_age: controls.max_age.map(duration).transpose()?,
        },
        scope: CacheScope::Isolated(scope(identity)),
    };
    Ok((
        body.into_iter()
            .filter(|(name, _)| !matches!(name.as_str(), "cache" | "caching"))
            .collect(),
        options,
    ))
}

fn duration(seconds: f64) -> Result<Duration, BodyError> {
    match Duration::try_from_secs_f64(seconds) {
        Ok(duration) if !duration.is_zero() => Ok(duration),
        Ok(_) => Err(BodyError::CacheDuration(None)),
        Err(error) => Err(BodyError::CacheDuration(Some(error))),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::test_support::{Held, key};

    #[rstest]
    #[case::the_master_key(Caller::MasterKey, r#"["litellm:master-key","master","master"]"#)]
    #[case::a_key_by_its_stored_hash(
        key("stored", false),
        r#"["litellm:virtual-key","stored","stored"]"#
    )]
    fn each_caller_is_served_only_its_own_cached_answers(
        #[case] caller: Caller<Held>,
        #[case] expected: &str,
    ) {
        assert_eq!(scope(&caller), expected);
    }
}
