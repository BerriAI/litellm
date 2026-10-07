use std::time::{Duration, SystemTime, UNIX_EPOCH};

use litellm_cache::{ExactCacheContext, SemanticCacheContext};
use litellm_cache_response::{CacheControls, CacheKeyField, CacheKeyInput, ResponseCacheRequest};
use litellm_host_python::from_py;
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestInput {
    key: CacheKeyInput,
    controls: Option<CacheControls>,
    ttl_seconds: Option<f64>,
    max_age_seconds: Option<f64>,
    messages: Option<Value>,
    input: Option<Value>,
    metadata: Option<Value>,
    litellm_metadata: Option<Value>,
    litellm_params: Option<Value>,
    scope: Option<String>,
}

#[derive(Clone)]
pub(super) struct NativeRequest {
    pub(super) key: CacheKeyInput,
    pub(super) controls: CacheControls,
    pub(super) ttl: Option<Duration>,
    pub(super) max_age: Option<Duration>,
    pub(super) messages: Option<Value>,
    pub(super) input: Option<Value>,
    pub(super) metadata: Option<Value>,
    pub(super) litellm_metadata: Option<Value>,
    pub(super) litellm_params: Option<Value>,
    pub(super) scope: Option<String>,
}

impl NativeRequest {
    pub(super) fn exact(&self) -> ResponseCacheRequest<ExactCacheContext> {
        ResponseCacheRequest {
            key: self.key.clone(),
            controls: self.controls,
            context: ExactCacheContext { ttl: self.ttl },
            max_age: self.max_age,
        }
    }

    /// The request as a semantic backend that keys on the caller's scope sees it.
    pub(super) fn semantic(&self) -> ResponseCacheRequest<SemanticCacheContext> {
        self.semantic_with(self.key.clone(), self.scope.clone())
    }

    /// The request keyed the way Python's Valkey semantic cache keys it: prompt fields drop out
    /// and the tenant identifiers for `scope` join the key.
    pub(super) fn scoped_semantic(
        &self,
        scope: &str,
    ) -> ResponseCacheRequest<SemanticCacheContext> {
        self.semantic_with(semantic_key(self, scope), Some(scope.to_owned()))
    }

    fn semantic_with(
        &self,
        key: CacheKeyInput,
        scope: Option<String>,
    ) -> ResponseCacheRequest<SemanticCacheContext> {
        ResponseCacheRequest {
            key,
            controls: self.controls,
            context: SemanticCacheContext {
                input: self.input.clone(),
                messages: self.messages.clone(),
                metadata: self.metadata.clone(),
                scope,
                ttl: self.ttl,
            },
            max_age: self.max_age,
        }
    }
}

fn semantic_key(request: &NativeRequest, scope: &str) -> CacheKeyInput {
    let mut key = request.key.clone();
    if key.preset.is_some() {
        return key;
    }
    key.fields
        .retain(|field| !matches!(field.name.as_str(), "messages" | "prompt" | "input"));
    const TENANT: [&str; 3] = [
        "user_api_key",
        "user_api_key_team_id",
        "user_api_key_org_id",
    ];
    let end_user = (scope == "end_user").then_some("user_api_key_end_user_id");
    for name in TENANT.into_iter().chain(end_user) {
        let sources = [
            request.metadata.as_ref(),
            request.litellm_metadata.as_ref(),
            request
                .litellm_params
                .as_ref()
                .and_then(|params| params.get("metadata")),
            request
                .litellm_params
                .as_ref()
                .and_then(|params| params.get("litellm_metadata")),
        ];
        let Some(value) = sources.into_iter().flatten().find_map(|source| {
            source
                .as_object()
                .and_then(|values| values.get(name))
                .filter(|value| !value.is_null())
        }) else {
            continue;
        };
        let value = match value {
            Value::Null => continue,
            Value::String(text) => text.clone(),
            other => other.to_string(),
        };
        key.fields.push(CacheKeyField {
            name: name.to_owned(),
            value: Some(value),
            api_parameter: true,
            internal_parameter: false,
        });
    }
    key
}

pub(super) fn request(value: &Bound<'_, PyAny>) -> PyResult<NativeRequest> {
    let input: RequestInput = from_py(value)?;
    request_input(input)
}

fn request_input(input: RequestInput) -> PyResult<NativeRequest> {
    let controls = input.controls.unwrap_or_else(|| {
        ResponseCacheRequest::<ExactCacheContext>::new(input.key.clone()).controls
    });
    Ok(NativeRequest {
        key: input.key,
        controls,
        ttl: input.ttl_seconds.map(duration).transpose()?,
        max_age: input.max_age_seconds.map(duration).transpose()?,
        messages: input.messages,
        input: input.input,
        metadata: input.metadata,
        litellm_metadata: input.litellm_metadata,
        litellm_params: input.litellm_params,
        scope: input.scope,
    })
}

pub(super) fn requests(value: &Bound<'_, PyAny>) -> PyResult<Vec<NativeRequest>> {
    from_py::<Vec<RequestInput>>(value)?
        .into_iter()
        .map(request_input)
        .collect()
}

pub(super) fn duration(seconds: f64) -> PyResult<Duration> {
    Duration::try_from_secs_f64(seconds)
        .map_err(|_| PyValueError::new_err("cache durations must be finite and nonnegative"))
}

pub(super) fn now() -> Duration {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use litellm_cache_response::{CacheControls, CacheKeyInput, cache_key};
    use serde_json::json;
    use sha2::{Digest, Sha256};

    use super::*;

    fn native_request(key: CacheKeyInput, metadata: Value) -> NativeRequest {
        NativeRequest {
            key,
            controls: CacheControls::default(),
            ttl: None,
            max_age: None,
            messages: Some(json!([{"role": "user", "content": "prompt"}])),
            input: None,
            metadata: Some(metadata),
            litellm_metadata: None,
            litellm_params: None,
            scope: None,
        }
    }

    #[test]
    fn semantic_key_matches_python_scope_material() {
        let key = CacheKeyInput {
            fields: vec![
                CacheKeyField {
                    name: "model".to_owned(),
                    value: Some("gpt-4.1".to_owned()),
                    api_parameter: true,
                    internal_parameter: false,
                },
                CacheKeyField {
                    name: "messages".to_owned(),
                    value: Some("prompt".to_owned()),
                    api_parameter: true,
                    internal_parameter: false,
                },
            ],
            ..Default::default()
        };
        let request = native_request(
            key,
            json!({"user_api_key": "k1", "user_api_key_team_id": null}),
        );
        let expected = format!("{:x}", Sha256::digest(b"model: gpt-4.1user_api_key: k1"));
        assert_eq!(cache_key(&semantic_key(&request, "key")), expected);
        assert_eq!(cache_key(&request.scoped_semantic("key").key), expected);

        let end_user_request = native_request(
            request.key.clone(),
            json!({"user_api_key": "k1", "user_api_key_end_user_id": "u1"}),
        );
        let expected = format!(
            "{:x}",
            Sha256::digest(b"model: gpt-4.1user_api_key: k1user_api_key_end_user_id: u1")
        );
        assert_eq!(
            cache_key(&semantic_key(&end_user_request, "end_user")),
            expected
        );

        let preset_request = native_request(
            CacheKeyInput {
                preset: Some("preset-key".to_owned()),
                ..Default::default()
            },
            json!({"user_api_key": "k1"}),
        );
        assert_eq!(
            semantic_key(&preset_request, "end_user").preset.as_deref(),
            Some("preset-key")
        );
        assert!(semantic_key(&preset_request, "end_user").fields.is_empty());
        assert_eq!(preset_request.semantic().context.scope, None);
        assert_eq!(
            preset_request
                .scoped_semantic("end_user")
                .context
                .scope
                .as_deref(),
            Some("end_user")
        );
    }
}
