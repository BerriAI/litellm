use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

use crate::call_type::CallTypes;
use crate::catalog::ModelInfoCatalog;
use crate::provider::LlmProviders;

static CHAT_SIZE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\d+)b").unwrap());
static EMBEDDING_SIZE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\d+)m").unwrap());

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TogetherThresholds {
    pub chat: [u64; 6],
    pub embedding: [u64; 2],
}

impl Default for TogetherThresholds {
    fn default() -> Self {
        Self {
            chat: [4, 8, 21, 41, 80, 110],
            embedding: [150, 350],
        }
    }
}

pub fn has_together_registry_pricing(model: &str, cost_map: &HashMap<String, Value>) -> bool {
    let stripped = model.strip_prefix("together_ai/").unwrap_or(model);
    cost_map
        .get(&format!("together_ai/{stripped}"))
        .and_then(Value::as_object)
        .is_some_and(|entry| entry.contains_key("input_cost_per_token"))
}

pub fn get_model_params_and_category(
    model_name: &str,
    call_type: &str,
    thresholds: TogetherThresholds,
) -> String {
    if matches!(
        call_type.parse::<CallTypes>(),
        Ok(CallTypes::embedding | CallTypes::aembedding)
    ) {
        return get_model_params_and_category_embeddings(model_name, thresholds);
    }
    let name = model_name.to_ascii_lowercase();
    let size = CHAT_SIZE
        .captures(&name)
        .and_then(|captures| captures.get(1)?.as_str().parse::<u64>().ok());
    let categories = [
        "together-ai-up-to-4b",
        "together-ai-4.1b-8b",
        "together-ai-8.1b-21b",
        "together-ai-21.1b-41b",
        "together-ai-41.1b-80b",
        "together-ai-81.1b-110b",
    ];
    size.and_then(|size| {
        thresholds
            .chat
            .iter()
            .position(|threshold| size <= *threshold)
            .map(|index| categories[index])
    })
    .unwrap_or(&name)
    .to_owned()
}

pub fn get_model_params_and_category_embeddings(
    model_name: &str,
    thresholds: TogetherThresholds,
) -> String {
    let name = model_name.to_ascii_lowercase();
    let size = EMBEDDING_SIZE
        .captures(&name)
        .and_then(|captures| captures.get(1)?.as_str().parse::<u64>().ok());
    let categories = [
        "together-ai-embedding-up-to-150m",
        "together-ai-embedding-151m-to-350m",
    ];
    size.and_then(|size| {
        thresholds
            .embedding
            .iter()
            .position(|threshold| size <= *threshold)
            .map(|index| categories[index])
    })
    .unwrap_or(&name)
    .to_owned()
}

pub fn together_pricing_model<'a>(
    catalog: &ModelInfoCatalog,
    model: &'a str,
    provider: Option<&str>,
    call_type: &str,
) -> Cow<'a, str> {
    let is_together = LlmProviders::TOGETHER_AI.matches(provider)
        || model.contains("togethercomputer")
        || model.contains("together_ai");
    if is_together && !has_together_registry_pricing(model, catalog.entries()) {
        Cow::Owned(get_model_params_and_category(
            model,
            call_type,
            TogetherThresholds::default(),
        ))
    } else {
        Cow::Borrowed(model)
    }
}
