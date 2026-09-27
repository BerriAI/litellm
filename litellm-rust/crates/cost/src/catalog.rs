use std::borrow::Cow;
use std::collections::HashMap;

use jiff::Timestamp;
use serde_json::Value;

use crate::completion_input::{
    CompletionInputRequest, PreparedCompletionInput, prepare_completion_input,
};
use crate::error::CostError;
use crate::fallback_generalizations::FallbackGeneralizations;
use crate::get_llm_provider::{LlmProvider, ProviderModelSets, get_llm_provider};
use crate::model_info::{ModelCost, ModelInfo, get_model_info_helper, lowercase_key_map};

const FALLBACK_GENERALIZATIONS_KEY: &str = "fallback_generalizations";
use crate::model_selection::{
    ModelSelectionRequest, get_provider_for_cost_calc, select_model_name_for_cost_calc,
};
use crate::responses_usage::ChatUsage;

#[derive(Clone, Debug, Default)]
pub struct ModelInfoCatalog {
    entries: HashMap<String, Value>,
    lowercase_keys: HashMap<String, String>,
    model_sets: ProviderModelSets,
    generalizations: FallbackGeneralizations,
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

fn cost_map_model<'m>(
    entries: &HashMap<String, Value>,
    model: &'m str,
    provider: Option<&str>,
    region: Option<&str>,
) -> Cow<'m, str> {
    let model = match provider {
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
            let bare = model.strip_prefix(&prefix).unwrap_or(model);
            let regional = region.map(|region| format!("{provider}/{region}/{bare}"));
            if let Some(regional) = regional.filter(|key| entries.contains_key(key)) {
                Cow::Owned(regional)
            } else if model.starts_with(&prefix) {
                Cow::Borrowed(model)
            } else {
                Cow::Owned(format!("{provider}/{model}"))
            }
        }
        None => Cow::Borrowed(model),
    };
    let without_prefix = model
        .split_once('/')
        .map_or(model, |(_, remainder)| remainder);
    if entries.contains_key(model_with_provider.as_ref()) {
        model_with_provider
    } else if entries.contains_key(model) {
        Cow::Borrowed(model)
    } else if entries.contains_key(without_prefix) {
        Cow::Borrowed(without_prefix)
    } else {
        Cow::Borrowed(model)
    }
}

impl ModelInfoCatalog {
    pub fn new(mut entries: HashMap<String, Value>) -> Self {
        let generalizations = FallbackGeneralizations::from_block(
            entries.remove(FALLBACK_GENERALIZATIONS_KEY).as_ref(),
        );
        Self {
            lowercase_keys: lowercase_key_map(&entries),
            model_sets: ProviderModelSets::from_model_cost(&entries),
            generalizations,
            entries,
        }
    }

    fn model_cost(&self) -> ModelCost<'_> {
        ModelCost {
            entries: &self.entries,
            lowercase_keys: &self.lowercase_keys,
            sets: &self.model_sets,
            generalizations: &self.generalizations,
        }
    }

    pub fn entries(&self) -> &HashMap<String, Value> {
        &self.entries
    }

    pub fn get_llm_provider(&self, model: &str) -> Option<LlmProvider> {
        get_llm_provider(model, &self.model_sets, &self.generalizations)
    }

    pub fn get_model_info(
        &self,
        model: &str,
        provider: Option<&str>,
    ) -> Result<ModelInfo<'_>, CostError> {
        get_model_info_helper(self.model_cost(), model, provider)
    }

    pub fn cost_map_model<'m>(
        &self,
        model: &'m str,
        provider: Option<&str>,
        region: Option<&str>,
    ) -> Cow<'m, str> {
        cost_map_model(&self.entries, model, provider, region)
    }

    pub fn select_model_info(
        &self,
        model: &str,
        provider: Option<&str>,
        region: Option<&str>,
    ) -> Option<ModelInfo<'_>> {
        let selected = cost_map_model(&self.entries, model, provider, region);
        let inferred = provider
            .is_none()
            .then(|| self.get_llm_provider(model))
            .flatten();
        let provider = provider.or(inferred
            .as_ref()
            .map(|inferred| inferred.custom_llm_provider.as_str()));
        self.get_model_info(&selected, provider).ok()
    }

    pub fn entry(
        &self,
        model: &str,
        provider: Option<&str>,
        region: Option<&str>,
    ) -> Option<Cow<'_, Value>> {
        self.select_model_info(model, provider, region)
            .map(|model_info| model_info.info)
    }

    pub fn contains_exact_model(&self, model: &str) -> bool {
        self.entries.contains_key(model)
    }

    pub fn select_model_name_for_cost_calc(
        &self,
        request: ModelSelectionRequest<'_>,
    ) -> Option<String> {
        select_model_name_for_cost_calc(request, self)
    }

    pub fn pricing_entry_for_cost_calc<'a>(
        &'a self,
        request: ModelSelectionRequest<'a>,
        logging_details: Option<&'a Value>,
    ) -> Option<ModelInfo<'a>> {
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
            return Some(ModelInfo {
                key: Cow::Borrowed(key),
                info: Cow::Borrowed(info),
            });
        }
        let selected = self.select_model_name_for_cost_calc(request);
        let provider = get_provider_for_cost_calc(request.model, request.provider, self);
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
        .find_map(|model| self.select_model_info(model, provider.as_deref(), None))
    }

    pub fn prepare_completion_input(
        &self,
        request: CompletionInputRequest<'_>,
    ) -> Result<PreparedCompletionInput, CostError> {
        prepare_completion_input(request, self)
    }
}
