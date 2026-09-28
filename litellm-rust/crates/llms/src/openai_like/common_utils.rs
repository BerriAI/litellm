//! Shared OpenAI-like credential and endpoint resolution, mirroring
//! `litellm/llms/openai_like/common_utils.py`.

use crate::Error;

/// `OpenAILikeChatConfig._get_openai_compatible_provider_info`: the deployment's
/// `api_base` wins over `OPENAI_LIKE_API_BASE`, and the deployment key over
/// `OPENAI_LIKE_API_KEY`, with an empty key allowed because vllm-compatible
/// endpoints do not require one.
pub fn openai_compatible_provider_info(
    api_base: Option<&str>,
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> (Option<String>, Option<String>) {
    let api_base = api_base
        .map(str::to_string)
        .or_else(|| env_lookup("OPENAI_LIKE_API_BASE"));
    let api_key = api_key
        .map(str::to_string)
        .or_else(|| env_lookup("OPENAI_LIKE_API_KEY"))
        .or(Some(String::new()));
    (api_base, api_key)
}

/// `OpenAILikeBase._validate_environment` requires an api base and, when the
/// caller gave no `custom_endpoint`, appends the route suffix. A caller-supplied
/// `custom_endpoint` base is used as is.
pub fn complete_openai_like_url(
    api_base: Option<&str>,
    custom_endpoint: bool,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<url::Url, Error> {
    let (api_base, _) = openai_compatible_provider_info(api_base, None, env_lookup);
    let api_base = api_base.ok_or_else(|| {
        Error::InvalidRequest(
            "Missing API Base - A call is being made to LLM Provider but no api base is set either in the environment variables ({LLM_PROVIDER}_API_KEY) or via params"
                .into(),
        )
    })?;
    let base = litellm_core_utils::url_utils::ApiUrl::parse(&api_base)?;
    if custom_endpoint {
        return Ok(base.into_url());
    }
    Ok(base.append_path(&["chat", "completions"])?.into_url())
}

/// The api key the call resolves to. `None` means neither the deployment nor the
/// environment supplied one, which is valid for endpoints that take no key.
pub fn resolve_openai_like_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    openai_compatible_provider_info(None, api_key, env_lookup)
        .1
        .filter(|key| !key.is_empty())
}
