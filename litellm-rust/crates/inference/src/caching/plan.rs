use std::{collections::BTreeMap, sync::Arc};

use litellm_cache_response::{CacheKeyInput, CacheOptions, ResponseCacheService};
use litellm_host::interceptors::WireRequest;
use serde_json::Value;

use super::CacheKeyProjection;
use crate::RouteError;

pub struct CachePlan {
    pub(super) service: Arc<dyn ResponseCacheService>,
    pub(super) options: CacheOptions,
    pub(super) input: CacheKeyInput,
}

impl CachePlan {
    pub fn new(
        service: Arc<dyn ResponseCacheService>,
        options: CacheOptions,
        input: CacheKeyInput,
    ) -> Option<Self> {
        options.policy.enabled().then_some(Self {
            service,
            options,
            input,
        })
    }

    pub(crate) fn for_request(
        service: Option<&Arc<dyn ResponseCacheService>>,
        options: Option<CacheOptions>,
        request: &impl CacheKeyProjection,
        model_group: Option<&str>,
    ) -> Result<Option<Self>, RouteError> {
        let (Some(service), Some(options)) = (service, options) else {
            return Ok(None);
        };
        if !options.policy.enabled() {
            return Ok(None);
        }
        Ok(Some(Self {
            service: service.clone(),
            options,
            input: request.cache_key_input(model_group)?,
        }))
    }

    pub(crate) fn guard(self, outbound: &WireRequest) -> GuardedPlan {
        GuardedPlan {
            sent: SentRequest::of(outbound),
            plan: self,
        }
    }
}

pub(crate) struct GuardedPlan {
    plan: CachePlan,
    sent: SentRequest,
}

impl GuardedPlan {
    pub(crate) fn confirm(self, wire: &WireRequest) -> Option<CachePlan> {
        if self.sent.matches(wire) {
            return Some(self.plan);
        }
        tracing::debug!("request changed before the provider call, skipping the response cache");
        None
    }
}

#[derive(PartialEq)]
struct SentRequest {
    url: String,
    headers: BTreeMap<String, String>,
    body: Value,
}

impl SentRequest {
    fn of(wire: &WireRequest) -> Self {
        Self {
            url: wire.url.clone(),
            headers: normalized_headers(&wire.headers),
            body: wire.body.clone(),
        }
    }

    fn matches(&self, wire: &WireRequest) -> bool {
        self.url == wire.url
            && self.body == wire.body
            && self.headers == normalized_headers(&wire.headers)
    }
}

fn normalized_headers(headers: &[(String, String)]) -> BTreeMap<String, String> {
    headers
        .iter()
        .map(|(name, value)| (name.to_ascii_lowercase(), value.clone()))
        .collect()
}
