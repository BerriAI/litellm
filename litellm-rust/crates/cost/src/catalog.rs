use std::collections::HashMap;

use jiff::Timestamp;
use serde_json::Value;

use crate::completion_input::{
    CompletionInputRequest, PreparedCompletionInput, prepare_completion_input,
};
use crate::error::CostError;
use crate::model_selection::{
    ModelSelectionRequest, get_provider_for_cost_calc, select_model_name_for_cost_calc,
};
use crate::responses_usage::ChatUsage;

#[derive(Clone, Debug, Default)]
pub struct ModelInfoCatalog {
    entries: HashMap<String, Value>,
}

#[derive(Clone, Copy, Debug)]
pub struct ModelCostRequest<'a> {
    pub model: &'a str,
    pub provider: Option<&'a str>,
    pub region: Option<&'a str>,
    pub usage: &'a ChatUsage,
    pub service_tier: Option<&'a str>,
    pub data_residency: Option<&'a str>,
    pub vertex_location: Option<&'a str>,
    pub at: Timestamp,
    pub response_time_ms: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
pub enum CostCall<'a> {
    Token {
        call_type: &'a str,
        prompt_characters: Option<f64>,
        completion_characters: Option<f64>,
        request_model: Option<&'a str>,
    },
    Speech {
        prompt_characters: Option<f64>,
    },
    Transcription {
        duration_seconds: f64,
    },
    Rerank {
        billed_units: Option<&'a Value>,
    },
    VectorStoreSearch {
        api_type: Option<&'a str>,
    },
    Search {
        number_of_queries: Option<u64>,
        optional_params: &'a Value,
    },
    Ocr {
        response: &'a Value,
        deployment_info: Option<&'a Value>,
    },
    Batch {
        deployment_info: Option<&'a Value>,
    },
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

fn select_model_key<'a>(
    entries: &'a HashMap<String, Value>,
    model: &str,
    provider: Option<&str>,
    region: Option<&str>,
) -> Option<&'a str> {
    let match_provider = provider.map(|provider| match provider {
        "vertex_ai_beta" => "vertex_ai",
        provider => provider,
    });
    let normalized = match provider {
        Some(provider) => {
            let prefix = format!("{provider}/");
            let mut name = model;
            while let Some(remainder) = name.strip_prefix(&prefix) {
                if !remainder.starts_with(&prefix) {
                    break;
                }
                name = remainder;
            }
            name
        }
        None => model,
    };
    let model_with_provider = match provider {
        Some(provider) => {
            let prefix = format!("{provider}/");
            let bare = normalized.strip_prefix(&prefix).unwrap_or(normalized);
            let regional = region.map(|region| format!("{provider}/{region}/{bare}"));
            if let Some(regional) = regional.filter(|key| entries.contains_key(key)) {
                regional
            } else if normalized.starts_with(&prefix) {
                normalized.to_owned()
            } else {
                format!("{provider}/{normalized}")
            }
        }
        None => normalized.to_owned(),
    };
    let without_prefix = normalized
        .split_once('/')
        .map_or(normalized, |(_, remainder)| remainder);
    [model_with_provider.as_str(), normalized, without_prefix]
        .into_iter()
        .find_map(|candidate| {
            entries
                .get_key_value(candidate)
                .filter(|(_, model_info)| check_provider_match(model_info, match_provider))
                .map(|(key, _)| key.as_str())
        })
}

impl ModelInfoCatalog {
    pub fn new(entries: HashMap<String, Value>) -> Self {
        Self { entries }
    }

    pub fn entries(&self) -> &HashMap<String, Value> {
        &self.entries
    }

    pub fn entry_for_key(&self, key: &str) -> &Value {
        self.entries
            .get(key)
            .expect("key came from select_model_key")
    }

    pub fn entry(
        &self,
        model: &str,
        provider: Option<&str>,
        region: Option<&str>,
    ) -> Option<&Value> {
        self.select_model_key(model, provider, region)
            .and_then(|key| self.entries.get(key))
    }

    pub fn select_model_key<'a>(
        &'a self,
        model: &str,
        provider: Option<&str>,
        region: Option<&str>,
    ) -> Option<&'a str> {
        select_model_key(&self.entries, model, provider, region)
    }

    pub fn contains_exact_model(&self, model: &str) -> bool {
        self.entries.contains_key(model)
    }

    pub fn get_provider_for_cost_calc(
        &self,
        model: Option<&str>,
        provider: Option<&str>,
        known_providers: &[&str],
    ) -> Option<String> {
        get_provider_for_cost_calc(model, provider, known_providers, &self.entries)
    }

    pub fn select_model_name_for_cost_calc(
        &self,
        request: ModelSelectionRequest<'_>,
    ) -> Option<String> {
        select_model_name_for_cost_calc(request, &self.entries)
    }

    pub fn pricing_entry_for_cost_calc<'a>(
        &'a self,
        request: ModelSelectionRequest<'a>,
        logging_details: Option<&'a Value>,
    ) -> Option<(&'a str, &'a Value)> {
        let deployment_info = if request.custom_pricing {
            request
                .router_model_id
                .and_then(|id| self.entries.get(id))
                .or_else(|| {
                    logging_details
                        .and_then(|details| details.get("litellm_params"))
                        .and_then(|params| {
                            params
                                .pointer("/metadata/model_info")
                                .or_else(|| params.pointer("/litellm_metadata/model_info"))
                        })
                })
        } else {
            None
        };
        if let (Some(key), Some(info)) =
            (request.router_model_id.or(request.model), deployment_info)
        {
            return Some((key, info));
        }
        let selected = self.select_model_name_for_cost_calc(request);
        let provider = get_provider_for_cost_calc(
            request.model,
            request.provider,
            request.known_providers,
            &self.entries,
        );
        [
            selected.as_deref(),
            request
                .response
                .and_then(|response| response.get("model"))
                .and_then(Value::as_str),
            request.model,
        ]
        .into_iter()
        .flatten()
        .find_map(|model| {
            self.select_model_key(model, provider.as_deref(), None)
                .and_then(|key| self.entries.get(key).map(|info| (key, info)))
        })
    }

    pub fn prepare_completion_input(
        &self,
        request: CompletionInputRequest<'_>,
    ) -> Result<PreparedCompletionInput, CostError> {
        prepare_completion_input(request, &self.entries)
    }
}
