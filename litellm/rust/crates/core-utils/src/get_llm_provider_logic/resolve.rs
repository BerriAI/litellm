use super::endpoint::endpoint_matches_api_base;
use super::provider_info::{Call, index_error};
use std::collections::HashSet;

use super::secrets::{Deferred, SecretReader, Secrets, Source, Value};
use super::tables::{ProviderTables, RoutingRules};

#[derive(Debug, Clone, Copy, Default)]
pub struct ProviderRequest<'a> {
    pub model: &'a str,
    pub custom_llm_provider: Option<&'a str>,
    pub api_base: Option<&'a str>,
    pub api_key: Option<&'a str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    pub model: String,
    pub custom_llm_provider: Option<String>,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
}

impl Resolved {
    pub(super) fn new(
        model: &str,
        custom_llm_provider: Option<&str>,
        api_key: Option<String>,
        api_base: Option<String>,
    ) -> Self {
        Self {
            model: model.to_owned(),
            custom_llm_provider: custom_llm_provider.map(str::to_owned),
            api_key,
            api_base,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderFailure {
    pub message: String,
    pub model: String,
    pub announce: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Resolution {
    Resolved(Resolved),
    Failed(ProviderFailure),
    Deferred,
}

#[derive(Debug)]
pub(super) enum Halt {
    Deferred,
    Raised(String),
    UnknownProvider,
}

impl From<Deferred> for Halt {
    fn from(_: Deferred) -> Self {
        Self::Deferred
    }
}

pub(super) fn checked(
    api_base: Option<Value>,
    dynamic_api_key: Option<Value>,
) -> Result<(Option<String>, Option<String>), Halt> {
    let api_base = match api_base {
        None => None,
        Some(Value::Text(text)) => Some(text),
        Some(flag) => {
            return Err(Halt::Raised(format!(
                "api base needs to be a string. api_base={flag}"
            )));
        }
    };
    let dynamic_api_key = match dynamic_api_key {
        None => None,
        Some(Value::Text(text)) => Some(text),
        Some(flag) => {
            return Err(Halt::Raised(format!(
                "dynamic_api_key needs to be a string. dynamic_api_key={flag}"
            )));
        }
    };
    Ok((api_base, dynamic_api_key))
}

fn first_segment(model: &str) -> &str {
    model.split_once('/').map_or(model, |(prefix, _)| prefix)
}

fn is_anthropic_text_model(model: &str) -> bool {
    model == "claude-2" || model == "claude-instant-1"
}

fn is_non_openai_azure_model(tables: &ProviderTables, model: &str) -> bool {
    model.split_once('/').is_some_and(|(_, name)| {
        tables.lists.cohere_chat_models.contains(name)
            || tables
                .lists
                .mistral_chat_models
                .contains(&format!("mistral/{name}"))
    })
}

fn special_model_provider(
    model: &str,
    provider: Option<&str>,
    via: &str,
    route: &'static str,
    renamed: impl Fn(&str) -> bool,
) -> Option<(String, &'static str)> {
    if provider == Some(via) && renamed(model) {
        return Some((model.to_owned(), route));
    }
    let (prefix, rest) = model.split_once('/')?;
    (prefix == via && renamed(rest)).then(|| (rest.to_owned(), route))
}

fn routing_generalization(tables: &ProviderTables, model: &str) -> Result<Option<String>, Halt> {
    if model.is_empty() {
        return Ok(None);
    }
    let RoutingRules::Compiled(rules) = &tables.routing_rules else {
        return Err(Halt::Deferred);
    };
    if !model.is_ascii() || model.contains('\n') {
        return Err(Halt::Deferred);
    }
    for (pattern, provider) in rules {
        if pattern.is_match(model).map_err(|_| Halt::Deferred)? {
            return Ok(Some(provider.clone()));
        }
    }
    Ok(None)
}

enum Listed {
    Provider(&'static str),
    Ai21,
    Unchanged,
}

fn listed_provider(tables: &ProviderTables, model: &str) -> Listed {
    let lists = &tables.lists;
    let any = |sets: &[&HashSet<String>]| sets.iter().any(|set| set.contains(model));
    if any(&[
        &lists.open_ai_chat_completion_models,
        &lists.openai_video_generation_models,
    ]) || model.contains("ft:gpt-3.5-turbo")
        || model.contains("ft:gpt-4")
        || model.starts_with("gpt-image")
    {
        return Listed::Provider("openai");
    }
    if lists.open_ai_text_completion_models.contains(model) {
        return Listed::Provider("text-completion-openai");
    }
    if lists.anthropic_models.contains(model) {
        return Listed::Provider(if is_anthropic_text_model(model) {
            "anthropic_text"
        } else {
            "anthropic"
        });
    }
    if any(&[&lists.cohere_models, &lists.cohere_embedding_models]) {
        return Listed::Provider("cohere");
    }
    if lists.cohere_chat_models.contains(model) {
        return Listed::Provider("cohere_chat");
    }
    let id_length = tables.replicate_model_name_with_id_length;
    if lists.replicate_models.contains(model)
        || (model.contains(':') && model.chars().count() > id_length)
    {
        let tagged = model
            .split(':')
            .nth(1)
            .is_some_and(|id| id.chars().count() == id_length);
        return if tagged || lists.replicate_models.contains(model) {
            Listed::Provider("replicate")
        } else {
            Listed::Unchanged
        };
    }
    let ordered: [(&[&HashSet<String>], Listed); 13] = [
        (&[&lists.openrouter_models], Listed::Provider("openrouter")),
        (&[&lists.maritalk_models], Listed::Provider("maritalk")),
        (
            &[
                &lists.vertex_chat_models,
                &lists.vertex_code_chat_models,
                &lists.vertex_text_models,
                &lists.vertex_code_text_models,
                &lists.vertex_language_models,
                &lists.vertex_embedding_models,
                &lists.vertex_vision_models,
                &lists.vertex_ai_image_models,
                &lists.vertex_ai_video_models,
            ],
            Listed::Provider("vertex_ai"),
        ),
        (&[&lists.ai21_chat_models, &lists.ai21_models], Listed::Ai21),
        (
            &[&lists.aleph_alpha_models],
            Listed::Provider("aleph_alpha"),
        ),
        (&[&lists.baseten_models], Listed::Provider("baseten")),
        (&[&lists.nlp_cloud_models], Listed::Provider("nlp_cloud")),
        (&[&lists.petals_models], Listed::Provider("petals")),
        (
            &[
                &lists.bedrock_models,
                &lists.bedrock_embedding_models,
                &lists.bedrock_converse_models,
            ],
            Listed::Provider("bedrock"),
        ),
        (&[&lists.watsonx_models], Listed::Provider("watsonx")),
        (
            &[&lists.open_ai_embedding_models],
            Listed::Provider("openai"),
        ),
        (&[&lists.empower_models], Listed::Provider("empower")),
        (
            &[&lists.gradient_ai_models],
            Listed::Provider("gradient_ai"),
        ),
    ];
    if let Some((_, listed)) = ordered.into_iter().find(|(sets, _)| any(sets)) {
        return listed;
    }
    if model == "*" {
        return Listed::Provider("openai");
    }
    const PREFIXED: &[(&str, &str)] = &[
        ("bytez/", "bytez"),
        ("gdc/", "gdc"),
        ("lemonade/", "lemonade"),
        ("heroku/", "heroku"),
        ("cometapi/", "cometapi"),
        ("oci/", "oci"),
        ("compactifai/", "compactifai"),
        ("ovhcloud/", "ovhcloud"),
        ("clarifai/", "clarifai"),
        ("amazon_nova", "amazon_nova"),
        ("sap/", "sap"),
    ];
    PREFIXED
        .iter()
        .find(|(prefix, _)| model.starts_with(prefix))
        .map_or(Listed::Unchanged, |(_, name)| Listed::Provider(name))
}

struct Resolver<'a> {
    request: ProviderRequest<'a>,
    tables: &'a ProviderTables,
    secrets: Secrets<'a>,
    model: String,
}

impl<'a> Resolver<'a> {
    fn call(&self) -> Call<'_> {
        Call {
            tables: self.tables,
            secrets: &self.secrets,
            api_base: self.request.api_base,
            api_key: self.request.api_key,
        }
    }

    fn litellm_proxy(&self) -> Result<Resolved, Halt> {
        let model = self
            .model
            .strip_prefix("litellm_proxy/")
            .unwrap_or(&self.model);
        let call = self.call();
        let api_base = call.secrets.first(&[
            Source::Given(self.request.api_base),
            Source::SecretStr("LITELLM_PROXY_API_BASE"),
        ])?;
        let api_key = call.secrets.first(&[
            Source::Given(self.request.api_key),
            Source::SecretStr("LITELLM_PROXY_API_KEY"),
        ])?;
        Ok(Resolved::new(
            model,
            Some("litellm_proxy"),
            api_key.map(|value| value.to_string()),
            api_base.map(|value| value.to_string()),
        ))
    }

    fn resolve(&mut self) -> Result<Resolved, Halt> {
        let api_base = self.request.api_base;
        if self.secrets.secret_bool("USE_LITELLM_PROXY")? == Some(true)
            || self.tables.use_litellm_proxy
        {
            return self.litellm_proxy();
        }
        if first_segment(&self.model) == "azure"
            && is_non_openai_azure_model(self.tables, &self.model)
        {
            return Ok(Resolved::new(
                &self.model,
                Some("openai"),
                None,
                api_base.map(str::to_owned),
            ));
        }
        let mut provider = self.request.custom_llm_provider.map(str::to_owned);
        if let Some((model, route)) = special_model_provider(
            &self.model,
            provider.as_deref(),
            "cohere",
            "cohere_chat",
            |model| self.tables.lists.cohere_chat_models.contains(model),
        ) {
            self.model = model;
            provider = Some(route.to_owned());
        }
        if let Some((model, route)) = special_model_provider(
            &self.model,
            provider.as_deref(),
            "anthropic",
            "anthropic_text",
            is_anthropic_text_model,
        ) {
            self.model = model;
            provider = Some(route.to_owned());
        }
        if let Some(custom) = provider.as_deref().filter(|custom| !custom.is_empty())
            && self.model.split('/').next() != Some(custom)
        {
            self.model = format!("{custom}/{}", self.model);
        }
        if provider.as_deref() == Some("openrouter")
            && let Some(remainder) = self.model.strip_prefix("openrouter/")
        {
            let model = if remainder.contains('/') {
                remainder
            } else {
                self.model.as_str()
            };
            return Ok(Resolved::new(
                model,
                Some("openrouter"),
                None,
                api_base.map(str::to_owned),
            ));
        }
        let prefix = first_segment(&self.model).to_owned();
        let has_slash = self.model.contains('/');
        if has_slash && self.tables.is_json_provider(&prefix) {
            return self.call().openai_compatible(&self.model);
        }
        let lists = &self.tables.lists;
        if lists.provider_list.contains(&prefix)
            && !lists.model_list_set.contains(&prefix)
            && has_slash
        {
            return self.call().openai_compatible(&self.model);
        }
        if lists.provider_list.contains(&prefix) {
            let rest = self.model.split_once('/').ok_or_else(index_error)?.1;
            return Ok(Resolved::new(
                rest,
                Some(&prefix),
                None,
                api_base.map(str::to_owned),
            ));
        }
        if has_slash {
            let Some(custom) = &self.tables.custom_providers else {
                return Err(Halt::Deferred);
            };
            if custom.contains(&prefix) {
                return Ok(Resolved::new(
                    first_rest(&self.model),
                    Some(&prefix),
                    None,
                    api_base.map(str::to_owned),
                ));
            }
        }
        if let Some(api_base) = api_base.filter(|base| !base.is_empty()) {
            for endpoint in &self.tables.openai_compatible_endpoints {
                if !endpoint_matches_api_base(endpoint, api_base)? {
                    continue;
                }
                let (provider, dynamic_api_key) =
                    match self.call().inferred_from_endpoint(endpoint, api_base)? {
                        Some((name, key)) => (Some(name.to_owned()), key),
                        None => (provider.clone(), None),
                    };
                let (_, dynamic_api_key) = checked(None, dynamic_api_key)?;
                return Ok(Resolved {
                    model: self.model.clone(),
                    custom_llm_provider: provider,
                    api_key: dynamic_api_key,
                    api_base: Some(api_base.to_owned()),
                });
            }
        }
        let (provider, api_base, dynamic_api_key) = match listed_provider(self.tables, &self.model)
        {
            Listed::Provider(name) => (Some(name.to_owned()), api_base.map(text), None),
            Listed::Unchanged => (provider, api_base.map(text), None),
            Listed::Ai21 => (
                Some("ai21_chat".to_owned()),
                self.secrets.first(&[
                    Source::Given(api_base),
                    Source::Secret("AI21_API_BASE"),
                    Source::Literal("https://api.ai21.com/studio/v1"),
                ])?,
                self.secrets.first(&[
                    Source::Given(self.request.api_key),
                    Source::Secret("AI21_API_KEY"),
                ])?,
            ),
        };
        let provider = match provider {
            Some(provider) if !provider.is_empty() => provider,
            _ => routing_generalization(self.tables, &self.model)?
                .filter(|provider| !provider.is_empty())
                .ok_or(Halt::UnknownProvider)?,
        };
        let (api_base, dynamic_api_key) = checked(api_base, dynamic_api_key)?;
        Ok(Resolved {
            model: self.model.clone(),
            custom_llm_provider: Some(provider),
            api_key: dynamic_api_key,
            api_base,
        })
    }
}

fn text(value: &str) -> Value {
    Value::Text(value.to_owned())
}

fn first_rest(model: &str) -> &str {
    model.split_once('/').map_or(model, |(_, rest)| rest)
}

pub fn get_llm_provider(
    request: ProviderRequest<'_>,
    tables: &ProviderTables,
    reader: &dyn SecretReader,
) -> Resolution {
    let mut resolver = Resolver {
        request,
        tables,
        secrets: Secrets::new(reader, tables.secret_manager),
        model: request.model.to_owned(),
    };
    match resolver.resolve() {
        Ok(resolved) => Resolution::Resolved(Resolved {
            model: tables
                .resolve_latest_alias(resolved.model, resolved.custom_llm_provider.as_deref()),
            ..resolved
        }),
        Err(Halt::Deferred) => Resolution::Deferred,
        Err(Halt::UnknownProvider) => Resolution::Failed(ProviderFailure {
            message: format!(
                "LLM Provider NOT provided. Pass in the LLM provider you are trying to call. You passed model={}\n Pass model as E.g. For 'Huggingface' inference endpoints pass in `completion(model='huggingface/starcoder',..)` Learn more: https://docs.litellm.ai/docs/providers",
                resolver.model
            ),
            model: resolver.model,
            announce: !tables.suppress_debug_info,
        }),
        Err(Halt::Raised(text)) => Resolution::Failed(ProviderFailure {
            message: format!(
                "GetLLMProvider Exception - {text}\n\noriginal model: {}",
                resolver.model
            ),
            model: resolver.model,
            announce: false,
        }),
    }
}
