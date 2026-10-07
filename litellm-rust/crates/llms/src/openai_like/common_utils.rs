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
) -> Result<String, Error> {
    let (api_base, _) = openai_compatible_provider_info(api_base, None, env_lookup);
    let api_base = api_base.ok_or_else(|| {
        Error::InvalidRequest(
            "Missing API Base - A call is being made to LLM Provider but no api base is set either in the environment variables ({LLM_PROVIDER}_API_KEY) or via params"
                .into(),
        )
    })?;
    if custom_endpoint {
        return Ok(api_base);
    }
    Ok(format!(
        "{}/chat/completions",
        api_base.trim_end_matches('/')
    ))
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

/// One `providers.json` entry, as Python's `SimpleProviderConfig` reads it for the routes
/// Rust serves. `cache_control_ttl` is the entry's `constraints.cache_control_ttl`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JsonProvider {
    pub base_url: &'static str,
    env_names: EnvNames,
    pub cache_control_ttl: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EnvNames {
    Key([&'static str; 1]),
    KeyAndBase([&'static str; 2]),
}

impl JsonProvider {
    pub const fn new(
        base_url: &'static str,
        api_key_env: &'static str,
        api_base_env: Option<&'static str>,
        cache_control_ttl: bool,
    ) -> Self {
        let env_names = match api_base_env {
            Some(api_base_env) => EnvNames::KeyAndBase([api_key_env, api_base_env]),
            None => EnvNames::Key([api_key_env]),
        };
        Self {
            base_url,
            env_names,
            cache_control_ttl,
        }
    }

    pub fn api_key_env(&self) -> &'static str {
        self.secret_names()[0]
    }

    pub fn api_base_env(&self) -> Option<&'static str> {
        self.secret_names().get(1).copied()
    }

    pub fn secret_names(&self) -> &[&'static str] {
        match &self.env_names {
            EnvNames::Key(names) => names,
            EnvNames::KeyAndBase(names) => names,
        }
    }

    pub fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Option<String> {
        non_blank(api_key)
            .map(str::to_string)
            .or_else(|| env_lookup(self.api_key_env()).filter(|key| !key.trim().is_empty()))
    }

    pub fn resolve_api_base(
        &self,
        api_base: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> String {
        non_blank(api_base)
            .map(str::to_string)
            .or_else(|| {
                self.api_base_env()
                    .and_then(env_lookup)
                    .filter(|base| !base.trim().is_empty())
            })
            .unwrap_or_else(|| self.base_url.to_string())
    }
}

pub fn non_blank(value: Option<&str>) -> Option<&str> {
    value.filter(|value| !value.trim().is_empty())
}
