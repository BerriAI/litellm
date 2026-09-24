use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Value, json};

use crate::bedrock_common_utils::{
    get_bedrock_base_model, is_bedrock_region, strip_bedrock_routing_prefix,
};
use crate::error::CostError;
use crate::fallback_generalizations::FallbackGeneralizations;
use crate::get_llm_provider::{
    ModelList, ProviderModelSets, declared_authenticating_provider, get_llm_provider,
};

const AZURE_LLMS: [(&str, &str); 7] = [
    ("gpt-35-turbo", "azure/gpt-35-turbo"),
    ("gpt-35-turbo-16k", "azure/gpt-35-turbo-16k"),
    ("gpt-35-turbo-instruct", "azure/gpt-35-turbo-instruct"),
    ("azure/gpt-41", "gpt-4.1"),
    ("azure/gpt-41-mini", "gpt-4.1-mini"),
    ("azure/gpt-41-nano", "gpt-4.1-nano"),
    ("ada", "azure/ada"),
];

static STABLE_VERTEX_VERSION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"-\d+$").expect("valid vertex version pattern"));
static DATED_SNAPSHOT_SUFFIX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"-\d{4}-\d{2}-\d{2}$").expect("valid snapshot pattern"));
static OPENAI_FINETUNE_SUFFIX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(:[^:]*){3}$").expect("valid finetune pattern"));

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PotentialModelNames {
    pub split_model: String,
    pub combined_model_name: String,
    pub stripped_model_name: String,
    pub combined_stripped_model_name: String,
    pub provider_prefixed_model_name: String,
    pub custom_llm_provider: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelInfo<'a> {
    pub key: Cow<'a, str>,
    pub info: Cow<'a, Value>,
}

#[derive(Clone, Copy, Debug)]
pub struct ModelCost<'a> {
    pub entries: &'a HashMap<String, Value>,
    pub lowercase_keys: &'a HashMap<String, String>,
    pub sets: &'a ProviderModelSets,
    pub generalizations: &'a FallbackGeneralizations,
}

pub fn strip_model_name(model: &str, custom_llm_provider: Option<&str>) -> String {
    match custom_llm_provider {
        Some("bedrock" | "bedrock_converse") => get_bedrock_base_model(model),
        Some("vertex_ai" | "gemini" | "databricks") => {
            STABLE_VERTEX_VERSION.replace(model, "").into_owned()
        }
        _ if model.contains("ft:") => OPENAI_FINETUNE_SUFFIX.replace(model, "").into_owned(),
        _ => DATED_SNAPSHOT_SUFFIX.replace(model, "").into_owned(),
    }
}

fn split_mantle_region_prefix(model: &str) -> &str {
    match model.split_once('/') {
        Some((head, tail)) if is_bedrock_region(head) => tail,
        _ => model,
    }
}

pub fn resolve_fireworks_resource_name(model: &str) -> String {
    let stripped = model.strip_prefix("fireworks_ai/").unwrap_or(model);
    if stripped.starts_with("accounts/") || stripped.starts_with("FW-") || stripped.contains('#') {
        stripped.to_owned()
    } else if stripped.starts_with("routers/") || stripped.starts_with("models/") {
        format!("accounts/fireworks/{stripped}")
    } else if stripped.ends_with("-fast") {
        format!("accounts/fireworks/routers/{stripped}")
    } else {
        format!("accounts/fireworks/models/{stripped}")
    }
}

pub fn get_potential_model_names(
    model_cost: ModelCost<'_>,
    model: &str,
    custom_llm_provider: Option<&str>,
) -> PotentialModelNames {
    let (split_model, combined, stripped, combined_stripped, provider_prefixed, provider) =
        match custom_llm_provider {
            None => {
                let inferred = get_llm_provider(model, model_cost.sets, model_cost.generalizations);
                let provider = inferred
                    .as_ref()
                    .map(|inferred| inferred.custom_llm_provider.clone());
                let stripped = strip_model_name(model, provider.as_deref());
                (
                    inferred.map_or_else(|| model.to_owned(), |inferred| inferred.model),
                    model.to_owned(),
                    stripped.clone(),
                    stripped,
                    model.to_owned(),
                    provider,
                )
            }
            Some(provider) => {
                let (split, combined, provider_prefixed) =
                    match model.strip_prefix(&format!("{provider}/")) {
                        Some(split) => (split, model.to_owned(), format!("{provider}/{model}")),
                        None => (
                            model,
                            format!("{provider}/{model}"),
                            format!("{provider}/{model}"),
                        ),
                    };
                let stripped = strip_model_name(split, Some(provider));
                let combined_stripped = format!("{provider}/{stripped}");
                (
                    split.to_owned(),
                    combined,
                    stripped,
                    combined_stripped,
                    provider_prefixed,
                    Some(provider.to_owned()),
                )
            }
        };
    let split_model = match provider.as_deref() {
        Some("bedrock" | "bedrock_converse") => {
            strip_bedrock_routing_prefix(&split_model).to_owned()
        }
        _ => split_model,
    };
    let (split_model, combined_stripped) = match provider.as_deref() {
        Some(provider @ "bedrock_mantle") => {
            let region_free = split_mantle_region_prefix(&split_model).to_owned();
            let combined_stripped = format!(
                "bedrock_mantle/{}",
                strip_model_name(&region_free, Some(provider))
            );
            (region_free, combined_stripped)
        }
        _ => (split_model, combined_stripped),
    };
    let provider_prefixed = match provider.as_deref() {
        Some("fireworks_ai") => format!(
            "fireworks_ai/{}",
            resolve_fireworks_resource_name(&split_model)
        ),
        _ => provider_prefixed,
    };
    PotentialModelNames {
        split_model,
        combined_model_name: combined,
        stripped_model_name: stripped,
        combined_stripped_model_name: combined_stripped,
        provider_prefixed_model_name: provider_prefixed,
        custom_llm_provider: provider,
    }
}

pub fn get_model_cost_key<'a>(model_cost: ModelCost<'a>, potential_key: &str) -> Option<&'a str> {
    model_cost
        .entries
        .get_key_value(potential_key)
        .map(|(key, _)| key.as_str())
        .or_else(|| {
            model_cost
                .lowercase_keys
                .get(&potential_key.to_lowercase())
                .map(String::as_str)
        })
}

pub fn lowercase_key_map(model_cost: &HashMap<String, Value>) -> HashMap<String, String> {
    model_cost.keys().fold(HashMap::new(), |mut map, key| {
        map.entry(key.to_lowercase())
            .and_modify(|existing: &mut String| {
                if key < existing {
                    existing.clone_from(key);
                }
            })
            .or_insert_with(|| key.clone());
        map
    })
}

fn vertex_ai_model_alias(model: &str, sets: &ProviderModelSets) -> Option<String> {
    let meta = format!("meta/{model}");
    if sets.contains(ModelList::VertexLlama3, &meta) {
        return Some(meta);
    }
    let latest = format!("{model}@latest");
    (sets.contains(ModelList::VertexMistral, &latest)
        || sets.contains(ModelList::VertexAi21, &latest))
    .then_some(latest)
}

fn get_model_info_from_generalization(
    model_cost: ModelCost<'_>,
    candidates: &[&str],
    custom_llm_provider: Option<&str>,
) -> Option<(String, Value)> {
    if candidates
        .iter()
        .any(|candidate| get_model_cost_key(model_cost, candidate).is_some())
    {
        return None;
    }
    candidates.iter().find_map(|candidate| {
        let mut info = model_cost.generalizations.match_capabilities(candidate)?;
        if let Some(provider) = custom_llm_provider {
            info.insert(
                "litellm_provider".to_owned(),
                Value::String(provider.to_owned()),
            );
        }
        Some(((*candidate).to_owned(), Value::Object(info)))
    })
}

pub fn get_model_info_helper<'a>(
    model_cost: ModelCost<'a>,
    model: &str,
    custom_llm_provider: Option<&str>,
) -> Result<ModelInfo<'a>, CostError> {
    let model = AZURE_LLMS
        .iter()
        .find(|(alias, _)| *alias == model)
        .map_or(model, |(_, target)| *target);
    let custom_llm_provider = custom_llm_provider.map(|provider| match provider {
        "vertex_ai_beta" => "vertex_ai",
        provider => provider,
    });
    let model = match custom_llm_provider {
        Some("vertex_ai") => {
            vertex_ai_model_alias(model, model_cost.sets).map_or(Cow::Borrowed(model), Cow::Owned)
        }
        _ => Cow::Borrowed(model),
    };
    let names = get_potential_model_names(
        model_cost,
        &model,
        custom_llm_provider.or_else(|| declared_authenticating_provider(&model)),
    );
    let provider = names.custom_llm_provider.as_deref();
    if provider == Some("huggingface") {
        return Ok(ModelInfo {
            key: Cow::Owned(model.into_owned()),
            info: Cow::Owned(json!({
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
                "litellm_provider": "huggingface",
                "mode": "chat"
            })),
        });
    }
    let candidates = [
        names.combined_model_name.as_str(),
        &model,
        &names.split_model,
        &names.combined_stripped_model_name,
        &names.stripped_model_name,
        &names.provider_prefixed_model_name,
    ];
    let mapped = candidates
        .iter()
        .filter_map(|candidate| get_model_cost_key(model_cost, candidate))
        .find_map(|key| {
            let info = &model_cost.entries[key];
            check_provider_match(info, provider).then_some(ModelInfo {
                key: Cow::Borrowed(key),
                info: Cow::Borrowed(info),
            })
        });
    if let Some(mapped) = mapped {
        return Ok(mapped);
    }
    get_model_info_from_generalization(model_cost, &candidates, provider)
        .map(|(key, info)| ModelInfo {
            key: Cow::Owned(key),
            info: Cow::Owned(info),
        })
        .ok_or(CostError::ModelNotFound)
}

pub fn check_provider_match(model_info: &Value, provider: Option<&str>) -> bool {
    let Some(provider) = provider.filter(|provider| !provider.is_empty()) else {
        return true;
    };
    let Some(entry_provider) = model_info.get("litellm_provider").and_then(Value::as_str) else {
        return true;
    };
    entry_provider == provider
        || (provider == "vertex_ai" && entry_provider.starts_with("vertex_ai"))
        || (provider == "fireworks_ai" && entry_provider.starts_with("fireworks_ai"))
        || (provider.starts_with("bedrock") && entry_provider.starts_with("bedrock"))
        || provider == "litellm_proxy"
        || (provider == "azure_ai" && matches!(entry_provider, "azure" | "openai"))
        || provider == "github"
}
