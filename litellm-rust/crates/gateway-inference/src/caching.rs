use std::time::Duration;

use litellm_cache_response::{CacheAccess, CacheCredential, CacheOptions, CachePolicy, CacheScope};
use litellm_gateway_auth::AuthenticatedRequest;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::Error;

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

pub(crate) fn prepare(
    identity: &AuthenticatedRequest,
    body: Map<String, Value>,
) -> Result<Prepared, Error> {
    let controls: Controls = match body.get("cache").filter(|value| !value.is_null()) {
        Some(value) => serde_json::from_value(value.clone())
            .map_err(|error| Error::InvalidBody(error.to_string()))?,
        None => Controls::default(),
    };
    let active = body
        .get("caching")
        .filter(|value| !value.is_null())
        .map(|value| serde_json::from_value::<bool>(value.clone()))
        .transpose()
        .map_err(|error| Error::InvalidBody(error.to_string()))?
        .unwrap_or(true);
    let caller = identity.caller();
    let options = CacheOptions {
        policy: CachePolicy {
            access: CacheAccess {
                reads: active && !controls.no_cache,
                writes: active && !controls.no_store,
            },
            ttl: controls.ttl.map(duration).transpose()?,
            max_age: controls.max_age.map(duration).transpose()?,
        },
        scope: CacheScope {
            credential: Some(CacheCredential::new(
                caller.principal().authority(),
                caller.principal().subject(),
                &caller.authentication().credential_id,
            )),
            model_group: Some(crate::request::model_name(&body)?.to_owned()),
        },
    };
    Ok((
        body.into_iter()
            .filter(|(name, _)| !matches!(name.as_str(), "cache" | "caching"))
            .collect(),
        options,
    ))
}

fn duration(seconds: f64) -> Result<Duration, Error> {
    Duration::try_from_secs_f64(seconds)
        .ok()
        .filter(|duration| !duration.is_zero())
        .ok_or_else(|| Error::InvalidBody("cache durations must be finite and positive".into()))
}

#[derive(Clone, Default)]
pub(crate) struct CacheHeaders(std::sync::Arc<std::sync::OnceLock<String>>);

impl litellm_host::interceptors::Interceptors<litellm_inference::RouteError> for CacheHeaders {
    async fn before_provider_request(
        &self,
        wire: litellm_host::interceptors::WireRequest,
        _: litellm_host::interceptors::RequestContext,
    ) -> Result<litellm_host::interceptors::WireRequest, litellm_inference::RouteError> {
        Ok(wire)
    }

    async fn after_provider_response(
        &self,
        _: litellm_host::interceptors::RawResponse,
    ) -> Result<(), litellm_inference::RouteError> {
        Ok(())
    }

    async fn result_ready(
        &self,
        facts: litellm_host::interceptors::ExecutionFacts,
    ) -> Result<(), litellm_inference::RouteError> {
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
            response.headers_mut().insert("x-litellm-cache-key", value);
        }
        response
    }
}
