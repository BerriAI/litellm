use serde_json::Value;

use crate::error::{CoreError, CoreResult};
use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

use super::transformation::{EmbeddingsAuthStrategy, EmbeddingsProviderConfig};
use super::types::{EmbeddingsRequest, ProviderEmbeddingsRequest};

pub(super) fn prepare_embeddings_call(
    request: EmbeddingsRequest<'_>,
) -> CoreResult<ProviderEmbeddingsRequest> {
    let provider_info = get_custom_llm_provider(request.model, request.custom_llm_provider)
        .or_else(|| {
            request
                .custom_llm_provider
                .map(|provider| CustomLlmProvider {
                    model: request.model,
                    custom_llm_provider: provider,
                })
        })
        .ok_or_else(|| {
            CoreError::InvalidProvider(
                "unable to resolve custom_llm_provider for embeddings request".to_string(),
            )
        })?;
    let model = provider_info.model.to_string();
    let provider = provider_info.custom_llm_provider;

    let config = embeddings_provider_config(provider)
        .ok_or_else(|| CoreError::InvalidProvider(provider.to_string()))?;
    let env_lookup = |key: &str| std::env::var(key).ok();

    let mut headers = Vec::new();

    let auth_strategy = config.auth_strategy();
    let api_key = config.resolve_api_key(request.api_key, &env_lookup)?;
    let auth_header = match auth_strategy {
        EmbeddingsAuthStrategy::Bearer => {
            ("authorization".to_string(), format!("Bearer {api_key}"))
        }
        EmbeddingsAuthStrategy::Header(name) => (name.to_string(), api_key),
    };
    headers.push(auth_header);

    for (name, value) in config.default_headers() {
        headers.push((name.to_string(), value.to_string()));
    }

    if let Some(extra_headers) = request.extra_headers {
        for (name, value) in extra_headers {
            if let Some(value_str) = value.as_str() {
                headers.push((name, value_str.to_string()));
            }
        }
    }

    let url = config.complete_url(request.api_base, &model, &env_lookup)?;
    
    let mut body = serde_json::json!({
        "model": model,
        "input": request.input,
    });
    
    if let Some(encoding_format) = request.encoding_format {
        body["encoding_format"] = Value::String(encoding_format);
    }
    
    if let Some(dimensions) = request.dimensions {
        body["dimensions"] = Value::Number(dimensions.into());
    }
    
    if let Some(user) = request.user {
        body["user"] = Value::String(user);
    }
    
    let transformed = config.transform_request(body)?;

    Ok(ProviderEmbeddingsRequest {
        provider: provider.to_string(),
        model,
        config,
        url,
        body: transformed,
        upstream_headers: headers,
        timeout: request.timeout,
    })
}

fn embeddings_provider_config(provider: &str) -> Option<&'static dyn EmbeddingsProviderConfig> {
    match provider {
        "openai" => Some(&OpenAiEmbeddingsConfig),
        "cohere" => Some(&CohereEmbeddingsConfig),
        "bedrock" => Some(&BedrockEmbeddingsConfig),
        _ => None,
    }
}

struct OpenAiEmbeddingsConfig;

impl EmbeddingsProviderConfig for OpenAiEmbeddingsConfig {
    fn complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        Ok(api_base
            .map(|base| format!("{}/embeddings", base.trim_end_matches('/')))
            .unwrap_or_else(|| "https://api.openai.com/v1/embeddings".to_string()))
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        api_key
            .map(|k| k.to_string())
            .or_else(|| env_lookup("OPENAI_API_KEY"))
            .ok_or_else(|| {
                CoreError::InvalidRequest("OpenAI API key not found".to_string())
            })
    }
}

struct CohereEmbeddingsConfig;

impl EmbeddingsProviderConfig for CohereEmbeddingsConfig {
    fn complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        Ok(api_base
            .map(|base| format!("{}/embed", base.trim_end_matches('/')))
            .unwrap_or_else(|| "https://api.cohere.ai/v1/embed".to_string()))
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        api_key
            .map(|k| k.to_string())
            .or_else(|| env_lookup("COHERE_API_KEY"))
            .ok_or_else(|| {
                CoreError::InvalidRequest("Cohere API key not found".to_string())
            })
    }

    fn auth_strategy(&self) -> EmbeddingsAuthStrategy {
        EmbeddingsAuthStrategy::Header("authorization")
    }

    fn transform_request(&self, mut request: Value) -> CoreResult<Value> {
        if let Some(input) = request.get("input").cloned() {
            match input {
                Value::String(s) => {
                    request["texts"] = Value::Array(vec![Value::String(s)]);
                }
                Value::Array(arr) => {
                    request["texts"] = Value::Array(arr);
                }
                _ => {}
            }
            request.as_object_mut().unwrap().remove("input");
        }
        Ok(request)
    }
}

struct BedrockEmbeddingsConfig;

impl EmbeddingsProviderConfig for BedrockEmbeddingsConfig {
    fn complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        let region = env_lookup("AWS_REGION").unwrap_or_else(|| "us-east-1".to_string());
        Ok(api_base
            .map(|base| format!("{}/model/{}/invoke", base.trim_end_matches('/'), model))
            .unwrap_or_else(|| {
                format!(
                    "https://bedrock-runtime.{}.amazonaws.com/model/{}/invoke",
                    region, model
                )
            }))
    }

    fn resolve_api_key(
        &self,
        _api_key: Option<&str>,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        Ok("bedrock-signing".to_string())
    }

    fn auth_strategy(&self) -> EmbeddingsAuthStrategy {
        EmbeddingsAuthStrategy::Header("x-amz-security-token")
    }
}
