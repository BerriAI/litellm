use std::{sync::Arc, time::Duration};

use litellm_cache::ExactCacheContext;
use serde_json::Value;

use crate::{
    CacheControls, CacheKeyContext, CacheKeyField, CacheKeyInput, CacheKeyParticipation,
    RequestRewrite, ResponseCacheRequest, ResponseCacheService,
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
    pub fn enabled(&self) -> bool {
        self.caching != Some(false) && !(self.no_cache && self.no_store)
    }
}

#[derive(Clone)]
pub struct CacheOptions {
    pub policy: CachePolicy,
    pub scope: CacheScope,
    pub key_context: CacheKeyContext,
    pub key_input: Option<CacheKeyInput>,
}

impl CacheOptions {
    pub fn new(scope: CacheScope) -> Self {
        Self {
            policy: CachePolicy::default(),
            scope,
            key_context: CacheKeyContext::default(),
            key_input: None,
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
        let selected = match self.key_input {
            Some(selected) => CacheKeyInput {
                rewritten_request: input.rewritten_request,
                ..selected
            },
            None => input,
        };
        let sharing_group = self
            .key_context
            .model_group
            .as_ref()
            .is_some_and(|group| !group.is_empty());
        let input = self.key_context.project(CacheKeyInput {
            transport: selected.transport.filter(|_| !sharing_group),
            ..selected
        });
        let isolated = matches!(self.scope, CacheScope::Isolated(_));
        let isolated_preset = isolated && input.preset.is_some();
        let rewrite = RequestRewrite::from(&input);
        let fields = match input.preset.as_ref().filter(|_| isolated) {
            Some(preset) => vec![CacheKeyField {
                name: "preset".into(),
                value: Some(preset.clone()),
                participation: CacheKeyParticipation::Always,
            }],
            None => input.fields,
        };
        let scope = match self.scope {
            CacheScope::Shared => String::new(),
            CacheScope::Isolated(scope) => serde_json::json!(["isolated", scope]).to_string(),
        };
        ResponseCacheRequest {
            rewrite,
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
                include_provider_parameters: input.include_provider_parameters,
                transport: input.transport.filter(|_| !isolated_preset),
                rewritten_request: input.rewritten_request.filter(|_| !isolated_preset),
                fields: [("surface", surface.to_owned()), ("scope", scope)]
                    .into_iter()
                    .map(|(name, value)| CacheKeyField {
                        name: name.into(),
                        value: Some(value),
                        participation: CacheKeyParticipation::Always,
                    })
                    .chain(fields)
                    .collect(),
            },
            controls: CacheControls {
                caching: self.policy.caching,
                no_cache: self.policy.no_cache,
                no_store: self.policy.no_store,
                ..CacheControls::enabled()
            },
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
    pub key_context: CacheKeyContext,
    pub key_input: Option<CacheKeyInput>,
}

impl ScopedCache {
    pub fn new(service: Arc<dyn ResponseCacheService>, scope: CacheScope) -> Self {
        Self {
            service,
            scope,
            key_context: CacheKeyContext::default(),
            key_input: None,
        }
    }

    pub fn with_key_input(
        self,
        key_input: Option<CacheKeyInput>,
        key_context: CacheKeyContext,
    ) -> Self {
        Self {
            key_input,
            key_context,
            ..self
        }
    }

    pub fn options(&self, policy: Option<CachePolicy>) -> CacheOptions {
        CacheOptions {
            policy: policy.unwrap_or_default(),
            scope: self.scope.clone(),
            key_context: self.key_context.clone(),
            key_input: self.key_input.clone(),
        }
    }
}
