use std::{sync::Arc, time::Duration};

use litellm_cache::ExactCacheContext;
use serde_json::Value;

use crate::{
    CacheAccess, CacheKeyField, CacheKeyInput, RequestRewrite, ResponseCacheRequest,
    ResponseCacheService,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CacheScope {
    Shared,
    Isolated(String),
}

#[derive(Clone, Copy, Default)]
pub struct CachePolicy {
    pub caching: Option<bool>,
    pub no_cache: bool,
    pub no_store: bool,
    pub ttl: Option<Duration>,
    pub max_age: Option<Duration>,
}

impl CachePolicy {
    pub fn access(&self) -> CacheAccess {
        let active = self.caching != Some(false);
        CacheAccess {
            reads: active && !self.no_cache,
            writes: active && !self.no_store,
        }
    }

    pub fn enabled(&self) -> bool {
        self.access() != CacheAccess::NONE
    }
}

#[derive(Clone)]
pub struct CacheOptions {
    pub policy: CachePolicy,
    pub scope: CacheScope,
}

impl CacheOptions {
    pub fn new(scope: CacheScope) -> Self {
        Self {
            policy: CachePolicy::default(),
            scope,
        }
    }

    pub fn request(self, namespace: &str, surface: &str, input: Value) -> ResponseCacheRequest {
        self.request_with_key(namespace, surface, CacheKeyInput::from_parameters(input))
    }

    pub fn request_with_key(
        self,
        namespace: &str,
        surface: &str,
        input: CacheKeyInput,
    ) -> ResponseCacheRequest {
        let isolated = matches!(self.scope, CacheScope::Isolated(_));
        let isolated_preset = isolated && input.preset.is_some();
        let rewrite = RequestRewrite::from(&input);
        let fields = match input.preset.as_ref().filter(|_| isolated) {
            Some(preset) => vec![CacheKeyField {
                name: "preset".into(),
                value: Some(preset.clone()),
            }],
            None => input.fields,
        };
        let scope = match &self.scope {
            CacheScope::Shared => String::new(),
            CacheScope::Isolated(scope) => serde_json::json!(["isolated", scope]).to_string(),
        };
        ResponseCacheRequest {
            rewrite,
            scope: self.scope,
            key: CacheKeyInput {
                namespace: Some(format!(
                    "{}:inference-v2",
                    input
                        .namespace
                        .as_deref()
                        .filter(|value| !value.is_empty())
                        .unwrap_or(namespace)
                )),
                preset: input.preset.filter(|_| !isolated),
                transport: input.transport.filter(|_| !isolated_preset),
                rewritten_request: input.rewritten_request.filter(|_| !isolated_preset),
                fields: [("surface", surface.to_owned()), ("scope", scope)]
                    .into_iter()
                    .map(|(name, value)| CacheKeyField {
                        name: name.into(),
                        value: Some(value),
                    })
                    .chain(fields)
                    .collect(),
            },
            access: self.policy.access(),
            context: ExactCacheContext {
                ttl: self.policy.ttl,
            },
            max_age: self.policy.max_age,
        }
    }
}

#[derive(Clone)]
pub struct ScopedCache {
    pub service: Arc<dyn ResponseCacheService>,
    pub scope: CacheScope,
}

impl ScopedCache {
    pub fn new(service: Arc<dyn ResponseCacheService>, scope: CacheScope) -> Self {
        Self { service, scope }
    }

    pub fn options(&self, policy: Option<CachePolicy>) -> CacheOptions {
        CacheOptions {
            policy: policy.unwrap_or_default(),
            scope: self.scope.clone(),
        }
    }
}
