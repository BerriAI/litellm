use serde_json::Value;

use crate::error::{CoreError, CoreResult};
use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

use super::transformation::{ImagesAuthStrategy, ImagesProviderConfig};
use super::types::{
    ImagesEditRequest, ImagesGenerationRequest, ProviderImagesEditRequest,
    ProviderImagesGenerationRequest,
};

pub(super) fn prepare_images_generation_call(
    request: ImagesGenerationRequest<'_>,
) -> CoreResult<ProviderImagesGenerationRequest> {
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
                "unable to resolve custom_llm_provider for images generation request".to_string(),
            )
        })?;
    let model = provider_info.model.to_string();
    let provider = provider_info.custom_llm_provider;

    let config = images_provider_config(provider)
        .ok_or_else(|| CoreError::InvalidProvider(provider.to_string()))?;
    let env_lookup = |key: &str| std::env::var(key).ok();

    let mut headers = Vec::new();

    let auth_strategy = config.auth_strategy();
    let api_key = config.resolve_api_key(request.api_key, &env_lookup)?;
    let auth_header = match auth_strategy {
        ImagesAuthStrategy::Bearer => ("authorization".to_string(), format!("Bearer {api_key}")),
        ImagesAuthStrategy::Header(name) => (name.to_string(), api_key),
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

    let url = config.complete_url(request.api_base, &model, false, &env_lookup)?;

    let mut body = serde_json::json!({
        "model": model,
        "prompt": request.prompt,
    });

    if let Some(n) = request.n {
        body["n"] = Value::Number(n.into());
    }

    if let Some(size) = request.size {
        body["size"] = Value::String(size);
    }

    if let Some(response_format) = request.response_format {
        body["response_format"] = Value::String(response_format);
    }

    if let Some(user) = request.user {
        body["user"] = Value::String(user);
    }

    let transformed = config.transform_generation_request(body)?;

    Ok(ProviderImagesGenerationRequest {
        _provider: provider.to_string(),
        model,
        config,
        url,
        body: transformed,
        upstream_headers: headers,
        timeout: request.timeout,
    })
}

pub(super) fn prepare_images_edit_call(
    request: ImagesEditRequest<'_>,
) -> CoreResult<ProviderImagesEditRequest> {
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
                "unable to resolve custom_llm_provider for images edit request".to_string(),
            )
        })?;
    let model = provider_info.model.to_string();
    let provider = provider_info.custom_llm_provider;

    let config = images_provider_config(provider)
        .ok_or_else(|| CoreError::InvalidProvider(provider.to_string()))?;
    let env_lookup = |key: &str| std::env::var(key).ok();

    let mut headers = Vec::new();

    let auth_strategy = config.auth_strategy();
    let api_key = config.resolve_api_key(request.api_key, &env_lookup)?;
    let auth_header = match auth_strategy {
        ImagesAuthStrategy::Bearer => ("authorization".to_string(), format!("Bearer {api_key}")),
        ImagesAuthStrategy::Header(name) => (name.to_string(), api_key),
    };
    headers.push(auth_header);

    // For edit requests, we need multipart/form-data
    headers.push((
        "content-type".to_string(),
        "multipart/form-data".to_string(),
    ));

    if let Some(extra_headers) = request.extra_headers {
        for (name, value) in extra_headers {
            if let Some(value_str) = value.as_str() {
                headers.push((name, value_str.to_string()));
            }
        }
    }

    let url = config.complete_url(request.api_base, &model, true, &env_lookup)?;

    let mut body = serde_json::json!({
        "model": model,
        "prompt": request.prompt,
    });

    if let Some(n) = request.n {
        body["n"] = Value::Number(n.into());
    }

    if let Some(size) = request.size {
        body["size"] = Value::String(size);
    }

    if let Some(response_format) = request.response_format {
        body["response_format"] = Value::String(response_format);
    }

    if let Some(user) = request.user {
        body["user"] = Value::String(user);
    }

    let transformed = config.transform_edit_request(body)?;

    Ok(ProviderImagesEditRequest {
        _provider: provider.to_string(),
        model,
        config,
        url,
        body: transformed,
        image: request.image,
        mask: request.mask,
        upstream_headers: headers,
        timeout: request.timeout,
    })
}

fn images_provider_config(provider: &str) -> Option<&'static dyn ImagesProviderConfig> {
    match provider {
        "openai" => Some(&OpenAiImagesConfig),
        "stability" => Some(&StabilityImagesConfig),
        _ => None,
    }
}

struct OpenAiImagesConfig;

impl ImagesProviderConfig for OpenAiImagesConfig {
    fn complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        is_edit: bool,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        let path = if is_edit {
            "/images/edits"
        } else {
            "/images/generations"
        };
        Ok(api_base
            .map(|base| format!("{}{}", base.trim_end_matches('/'), path))
            .unwrap_or_else(|| format!("https://api.openai.com/v1{}", path)))
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        api_key
            .map(|k| k.to_string())
            .or_else(|| env_lookup("OPENAI_API_KEY"))
            .ok_or_else(|| CoreError::InvalidRequest("OpenAI API key not found".to_string()))
    }
}

struct StabilityImagesConfig;

impl ImagesProviderConfig for StabilityImagesConfig {
    fn complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        is_edit: bool,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        let path = if is_edit {
            format!("/v1/{}/image-to-image", model)
        } else {
            format!("/v1/{}/text-to-image", model)
        };
        Ok(api_base
            .map(|base| format!("{}{}", base.trim_end_matches('/'), path))
            .unwrap_or_else(|| format!("https://api.stability.ai{}", path)))
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        api_key
            .map(|k| k.to_string())
            .or_else(|| env_lookup("STABILITY_API_KEY"))
            .ok_or_else(|| CoreError::InvalidRequest("Stability API key not found".to_string()))
    }

    fn auth_strategy(&self) -> ImagesAuthStrategy {
        ImagesAuthStrategy::Header("authorization")
    }
}
