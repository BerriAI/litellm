use crate::index::{ModelIndex, parse_index};
use crate::providers::{known_provider, provider_names};
use crate::{Error, canonical_provider};
use serde_json::{Map, Value};
use std::collections::HashSet;
use std::sync::LazyLock;

#[derive(Debug)]
pub struct PricingCatalog {
    index: ModelIndex<Map<String, Value>>,
    providers: HashSet<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct PricingMatch<'a> {
    pub matched_key: &'a str,
    pub canonical_key: &'a str,
    pub fields: &'a Map<String, Value>,
}

impl PricingCatalog {
    pub fn parse(body: &[u8]) -> Result<Self, Error> {
        let parsed = parse_index(body, Ok)?;
        let providers = parsed
            .index
            .entries
            .values()
            .filter_map(|entry| entry.get("litellm_provider").and_then(Value::as_str))
            .map(str::to_owned)
            .collect();
        Ok(Self {
            index: parsed.index,
            providers,
        })
    }

    pub fn lookup(&self, key: &str) -> Option<PricingMatch<'_>> {
        let matched = self.index.lookup(key)?;
        Some(PricingMatch {
            matched_key: matched.matched_key,
            canonical_key: matched.canonical_key,
            fields: matched.entry,
        })
    }

    pub fn resolve(&self, model: &str, providers: &[&str]) -> Option<PricingMatch<'_>> {
        if model.is_empty() || providers.iter().any(|provider| provider.is_empty()) {
            return None;
        }
        let Some(first) = providers.first() else {
            return self.resolve_provider(model, None);
        };
        if providers
            .iter()
            .any(|provider| canonical_provider(provider) != canonical_provider(first))
        {
            return None;
        }
        let candidates = provider_names(first, model);
        let mut matches = candidates.iter().filter_map(|candidate| {
            providers
                .iter()
                .skip(1)
                .all(|provider| provider_names(provider, model).contains(candidate))
                .then(|| self.resolve_provider(model, Some(candidate)))
                .flatten()
        });
        let matched = matches.next()?;
        matches
            .all(|other| other.canonical_key == matched.canonical_key)
            .then_some(matched)
    }

    /// Pricing requirements carried by an observed identity, independently of served token rates.
    /// None means a recognized route has no catalog contract; an empty list is an opaque alias.
    pub fn request_contexts(
        &self,
        model: &str,
        providers: &[&str],
    ) -> Option<Vec<PricingMatch<'_>>> {
        let exact = self
            .resolve(model, providers)
            .or_else(|| self.resolve(model, &[]));
        let route_key = self.router_context_key(model, providers).or_else(|| {
            let matched = exact?;
            let provider = matched.fields.get("litellm_provider")?.as_str()?;
            self.router_context_key(matched.canonical_key, &[provider])
        });
        let route_key =
            route_key.or_else(|| self.router_context_key(exact?.canonical_key, providers));
        let route = match route_key {
            Some(key) => Some(self.lookup(key)?),
            None => None,
        };
        Some([exact, route].into_iter().flatten().collect())
    }

    fn router_context_key(&self, model: &str, providers: &[&str]) -> Option<&'static str> {
        if model.contains("agents/") {
            return None;
        }
        let model = model.to_lowercase();
        if !model.contains("model-router") && !model.contains("model_router") {
            return None;
        }
        let prefix = model
            .split_once('/')
            .map(|(prefix, _)| prefix)
            .filter(|prefix| {
                known_provider(prefix)
                    || self.providers.contains(*prefix)
                    || matches!(*prefix, "azure.ai.inference" | "azure.ai.openai")
            });
        prefix
            .into_iter()
            .chain(providers.iter().copied().filter(|_| prefix.is_none()))
            .find_map(|provider| match canonical_provider(provider) {
                "azure.ai.inference" => Some("azure_ai/model_router"),
                "azure.ai.openai" => Some("azure/model-router"),
                _ => None,
            })
    }

    fn resolve_provider(&self, model: &str, provider: Option<&str>) -> Option<PricingMatch<'_>> {
        let model = azure_model_name(model);
        let provider = provider
            .filter(|value| !value.is_empty())
            .or_else(|| {
                let prefix = model.split_once('/')?.0;
                (known_provider(prefix) || self.providers.contains(prefix)).then_some(prefix)
            })
            .map(|value| {
                if value == "vertex_ai_beta" {
                    "vertex_ai"
                } else {
                    value
                }
            });
        let model = self.vertex_model_name(model, provider);
        let split = provider
            .and_then(|provider| model.strip_prefix(&format!("{provider}/")))
            .unwrap_or(&model);
        let combined = provider.map_or_else(
            || model.clone(),
            |provider| {
                if model.starts_with(&format!("{provider}/")) {
                    model.clone()
                } else {
                    format!("{provider}/{model}")
                }
            },
        );
        let stripped = strip_model_name(split, provider);
        let combined_stripped = provider.map_or_else(
            || stripped.clone(),
            |provider| format!("{provider}/{stripped}"),
        );
        let split = match provider {
            Some("bedrock" | "bedrock_converse") => strip_bedrock_routes(split),
            Some("bedrock_mantle") => strip_region(split),
            _ => split,
        };
        let region_free = if provider == Some("bedrock_mantle") {
            format!("bedrock_mantle/{split}")
        } else {
            combined.clone()
        };
        let provider_prefixed = match provider {
            Some("fireworks_ai") => format!(
                "fireworks_ai/{}",
                if split.starts_with("accounts/") {
                    split.to_owned()
                } else {
                    format!("accounts/fireworks/models/{split}")
                }
            ),
            Some(provider) => format!("{provider}/{model}"),
            None => model.clone(),
        };
        [
            combined.as_str(),
            model.as_str(),
            region_free.as_str(),
            split,
            combined_stripped.as_str(),
            stripped.as_str(),
            provider_prefixed.as_str(),
        ]
        .into_iter()
        .filter_map(|key| self.lookup(key))
        .find(|matched| provider_matches(matched.fields, provider))
    }

    fn vertex_model_name(&self, model: &str, provider: Option<&str>) -> String {
        if provider != Some("vertex_ai") {
            return model.to_owned();
        }
        [format!("meta/{model}"), format!("{model}@latest")]
            .into_iter()
            .find(|candidate| {
                self.lookup(candidate)
                    .is_some_and(|matched| provider_matches(matched.fields, provider))
            })
            .unwrap_or_else(|| model.to_owned())
    }
}

fn provider_matches(fields: &Map<String, Value>, provider: Option<&str>) -> bool {
    let Some(provider) = provider else {
        return true;
    };
    let Some(value) = fields
        .get("litellm_provider")
        .filter(|value| !value.is_null())
    else {
        return true;
    };
    let Some(actual) = value.as_str() else {
        return false;
    };
    actual == provider
        || matches!(provider, "litellm_proxy" | "github" | "nadir")
        || (matches!(provider, "vertex_ai" | "fireworks_ai") && actual.starts_with(provider))
        || (provider.starts_with("bedrock") && actual.starts_with("bedrock"))
        || (provider == "azure_ai" && matches!(actual, "azure" | "openai"))
}

fn azure_model_name(model: &str) -> &str {
    match model {
        "gpt-35-turbo" => "azure/gpt-35-turbo",
        "gpt-35-turbo-16k" => "azure/gpt-35-turbo-16k",
        "gpt-35-turbo-instruct" => "azure/gpt-35-turbo-instruct",
        "azure/gpt-41" => "gpt-4.1",
        "azure/gpt-41-mini" => "gpt-4.1-mini",
        "azure/gpt-41-nano" => "gpt-4.1-nano",
        "ada" => "azure/ada",
        _ => model,
    }
}

fn strip_model_name(model: &str, provider: Option<&str>) -> String {
    if matches!(provider, Some("bedrock" | "bedrock_converse")) {
        return bedrock_base_model(model);
    }
    if provider == Some("bedrock_mantle") {
        return strip_region(model).to_owned();
    }
    if matches!(provider, Some("vertex_ai" | "gemini" | "databricks")) {
        return model
            .rsplit_once('-')
            .filter(|(_, suffix)| {
                !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
            })
            .map_or(model, |(base, _)| base)
            .to_owned();
    }
    if model.contains("ft:") {
        let mut parts = model.rsplitn(4, ':');
        return parts.nth(3).unwrap_or(model).to_owned();
    }
    let suffix = model.as_bytes().get(model.len().saturating_sub(11)..);
    if let Some(suffix) = suffix
        && suffix.len() == 11
        && [0, 5, 8].into_iter().all(|index| suffix[index] == b'-')
        && suffix
            .iter()
            .enumerate()
            .all(|(index, byte)| matches!(index, 0 | 5 | 8) || byte.is_ascii_digit())
    {
        return model[..model.len() - 11].to_owned();
    }
    model.to_owned()
}

fn strip_bedrock_routes(mut model: &str) -> &str {
    for prefix in [
        "bedrock/",
        "chat_completions/",
        "converse/",
        "invoke/",
        "openai/",
        "mantle/",
        "nova-2/",
        "nova/",
    ] {
        if let Some(stripped) = model.strip_prefix(prefix) {
            model = stripped;
        }
    }
    model
}

fn strip_region(model: &str) -> &str {
    model.split_once('/').map_or(model, |(prefix, rest)| {
        let parts: Vec<_> = prefix.split('-').collect();
        if parts.len() >= 3
            && matches!(
                parts[0],
                "us" | "eu" | "ap" | "ca" | "sa" | "me" | "af" | "il" | "mx" | "cn"
            )
            && parts.last().is_some_and(|part| part.parse::<u8>().is_ok())
        {
            rest
        } else {
            model
        }
    })
}

fn bedrock_base_model(model: &str) -> String {
    let route_free = ["bedrock/converse/", "bedrock/", "converse/"]
        .into_iter()
        .find_map(|prefix| model.strip_prefix(prefix))
        .unwrap_or(model);
    if route_free.starts_with("nova-2/") {
        return "amazon.nova-2-custom".to_owned();
    }
    if route_free.starts_with("nova/") {
        return "amazon.nova-custom".to_owned();
    }
    let model = strip_bedrock_routes(model);
    let model = if model.to_lowercase().contains("arn") {
        model.rsplit('/').next().unwrap_or(model)
    } else {
        model
    };
    let model = model.rsplit_once(':').map_or(model, |(base, suffix)| {
        if suffix
            .strip_suffix('k')
            .is_some_and(|value| value.parse::<u64>().is_ok())
        {
            base
        } else {
            model
        }
    });
    let model = model.split_once('.').map_or(model, |(region, rest)| {
        if matches!(
            region,
            "global" | "us" | "eu" | "apac" | "jp" | "au" | "us-gov"
        ) {
            rest
        } else {
            model
        }
    });
    strip_region(model).to_owned()
}

pub fn bundled_pricing_catalog() -> Result<&'static PricingCatalog, &'static Error> {
    static CATALOG: LazyLock<Result<PricingCatalog, Error>> = LazyLock::new(|| {
        PricingCatalog::parse(include_bytes!(
            "../../../../model_prices_and_context_window.json"
        ))
    });
    CATALOG.as_ref()
}
