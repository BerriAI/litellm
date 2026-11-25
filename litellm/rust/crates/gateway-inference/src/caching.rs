use std::time::Duration;

use litellm_cache_response::{CacheOptions, CachePolicy, CacheScope};
use litellm_gateway_auth::Authenticated;
use litellm_gateway_auth_core::{Caller, Rows, StoredKey};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::{BodyError, ExposedHeader};

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

#[derive(Clone, Default)]
pub(crate) struct CacheHeaders(std::sync::Arc<std::sync::OnceLock<String>>);

impl litellm_host::interceptors::Interceptors<litellm_inference::Error> for CacheHeaders {
    async fn before_provider_request(
        &self,
        wire: litellm_host::interceptors::WireRequest,
        _: litellm_host::interceptors::RequestContext,
    ) -> Result<litellm_host::interceptors::WireRequest, litellm_inference::Error> {
        Ok(wire)
    }

    async fn after_provider_response(
        &self,
        _: litellm_host::interceptors::RawResponse,
    ) -> Result<(), litellm_inference::Error> {
        Ok(())
    }

    async fn result_ready(
        &self,
        facts: litellm_host::interceptors::ExecutionFacts,
    ) -> Result<(), litellm_inference::Error> {
        if let litellm_host::interceptors::ResultSource::Cache { key } = facts.source {
            let _ = self.0.set(key);
        }
        Ok(())
    }
}

impl CacheHeaders {
    pub(crate) fn apply(&self, mut response: axum::response::Response) -> axum::response::Response {
        if let Some(key) = self.0.get()
            && let Ok(value) = axum::http::HeaderValue::from_str(key)
        {
            response
                .headers_mut()
                .insert(axum::http::HeaderName::from(ExposedHeader::CacheKey), value);
        }
        response
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::test_support::{Held, key};

    #[rstest]
    #[case::the_master_key(Caller::master_key(()), r#"["litellm:master-key","master","master"]"#)]
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
