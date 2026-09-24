use std::collections::HashMap;

use jiff::Timestamp;
use serde_json::Value;

use crate::anthropic_cost::fast_speed_multiplier;
use crate::azure_ai_cost::azure_ai_cost_per_token;
use crate::azure_ai_image_cost::{
    AzureAiImageRequest, cost_calculator as azure_ai_image_cost_calculator,
};
use crate::azure_cost::output_per_second_cost;
use crate::base_rate_selection::uses_inclusive_token_thresholds;
use crate::batch::batch_cost_from_model_info;
use crate::bedrock_image_cost::cost_calculator as bedrock_image_cost_calculator;
use crate::billed_token_rates::{
    BilledRatesRequest, BilledTokenRates, TokenTypeCostBreakdown,
    get_billed_token_rates as calculate_billed_token_rates,
    get_token_type_cost_breakdown as calculate_token_type_cost_breakdown,
};
use crate::completion_cost::{
    CompletionCost, completion_cost, get_response_cost_from_hidden_params,
};
use crate::completion_input::{
    CompletionInputRequest, PreparedCompletionInput, prepare_completion_input,
};
use crate::custom_pricing::{CustomPricing, CustomTokenRates, cost_from_chat_usage};
use crate::dashscope_cost::cost_per_token as dashscope_cost_per_token;
use crate::databricks_cost::databricks_cost_per_token;
use crate::error::CostError;
use crate::fal_ai_image_cost::{
    cost_calculator as fal_ai_image_cost_calculator,
    fal_ai_passthrough_cost as calculate_fal_ai_passthrough_cost,
};
use crate::fireworks_cost::{
    FireworksThresholds, cost_per_token as fireworks_cost_per_token, get_base_model_for_pricing,
};
use crate::generic_cost::calculate_generic_cost_from_model_info_with_region;
use crate::image_response_cost::{
    calculate_image_response_cost_from_usage, flat_image_cost, gemini_image_edit_cost,
    gemini_image_generation_cost, resolve_image_model_info, vertex_image_edit_cost,
    vertex_image_generation_cost,
};
use crate::lemonade_cost::lemonade_cost_per_token;
use crate::model_selection::{
    ModelSelectionRequest, get_provider_for_cost_calc, select_model_name_for_cost_calc,
};
use crate::non_token::{ImageRates, ImageUsage, calculate_image};
use crate::ocr_cost::ocr_cost;
use crate::openai_cost::video_generation_cost as calculate_video_generation_cost;
use crate::openai_image_cost::cost_calculator as openai_image_cost_calculator;
use crate::per_second::{has_token_or_tiered_pricing, per_second_pricing_cost};
use crate::perplexity_cost::cost_per_token as perplexity_cost_per_token;
use crate::prompt_caching_savings::{
    PromptCachingSavingsRequest, calculate_prompt_caching_savings,
};
use crate::provider::LlmProviders;
use crate::provider_cache::apply_provider_cache_read_default;
use crate::realtime_cost::{
    combine_usage_objects, event_usage, get_transcription_model_name_from_results,
    partition_results_by_service_tier, transcription_usage_cost,
};
use crate::regional_uplift::get_provider_specific_geo_multiplier;
use crate::responses_usage::ChatUsage;
use crate::retrieval_cost::{rerank_cost, vector_store_search_cost};
use crate::search_cost::{
    ParallelAiPricing, effective_mode, parallel_ai_search_cost, provider_usage,
    search_provider_cost_per_query,
};
use crate::speech_cost::{
    SpeechCostMetric, cost_per_second, generic_cost_per_character, lyria_generation_cost,
    select_cost_metric_for_model, transcription_usage_has_token_details,
};
use crate::together_cost::{
    TogetherThresholds, get_model_params_and_category, has_together_registry_pricing,
    together_ai_cost_per_token,
};
use crate::tool_cost_dispatch::{BuiltInToolCostRequest, get_cost_for_built_in_tools};
use crate::vertex_cost::{cost_per_token as vertex_cost_per_token, vertex_cost};
use crate::xai_cost::{cost_per_token as xai_cost_per_token, reported_cost as xai_reported_cost};

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

#[derive(Clone, Copy, Debug)]
pub struct CompletionCostRequest<'a> {
    pub token: ModelCostRequest<'a>,
    pub built_in_tools: BuiltInToolCharge<'a>,
    pub additional_costs: &'a [f64],
    pub discount_config: &'a Value,
    pub margin_config: &'a Value,
}

#[derive(Clone, Copy, Debug)]
pub enum BuiltInToolCharge<'a> {
    Provided(f64),
    FromResponse(BuiltInToolCostRequest<'a>),
}

#[derive(Clone, Copy, Debug)]
pub struct ResponseCostRequest<'a> {
    pub completion: CompletionCostRequest<'a>,
    pub cache_hit: bool,
    pub hidden_params: &'a Value,
}

#[derive(Clone, Copy, Debug)]
pub struct AzureAiImageCatalogRequest<'a> {
    pub model: &'a str,
    pub image_response: &'a Value,
    pub size: Option<&'a str>,
    pub n: Option<u64>,
    pub optional_params: &'a Value,
    pub supplied_model_info: Option<&'a Value>,
    pub at: Timestamp,
}

#[derive(Clone, Copy, Debug)]
pub struct DefaultImageCostRequest<'a> {
    pub model: &'a str,
    pub provider: Option<&'a str>,
    pub quality: Option<&'a str>,
    pub n: Option<u64>,
    pub size: Option<&'a str>,
    pub supplied_model_info: Option<&'a Value>,
}

fn select_model_key<'a, T>(
    entries: &'a HashMap<String, T>,
    model: &str,
    provider: Option<&str>,
    region: Option<&str>,
) -> Option<&'a str> {
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

    pub fn cost_per_token_with_custom(
        &self,
        request: ModelCostRequest<'_>,
        pricing: CustomPricing,
    ) -> Result<(f64, f64), CostError> {
        match cost_from_chat_usage(request.usage, pricing, request.response_time_ms)? {
            Some(cost) => Ok((cost.input, cost.output)),
            None => self.cost_per_token(request),
        }
    }

    pub fn cost_per_token_for_call_with_custom(
        &self,
        request: ModelCostRequest<'_>,
        call: CostCall<'_>,
        pricing: CustomPricing,
    ) -> Result<(f64, f64), CostError> {
        match cost_from_chat_usage(request.usage, pricing, request.response_time_ms)? {
            Some(cost) => Ok((cost.input, cost.output)),
            None => self.cost_per_token_for_call(request, call),
        }
    }

    pub fn cost_per_token_for_call(
        &self,
        request: ModelCostRequest<'_>,
        call: CostCall<'_>,
    ) -> Result<(f64, f64), CostError> {
        let provider = request.provider.and_then(LlmProviders::parse);
        match call {
            CostCall::Token {
                call_type,
                prompt_characters,
                completion_characters,
                request_model,
            } => {
                if provider == Some(LlmProviders::VERTEX_AI) {
                    return vertex_cost(
                        self,
                        request,
                        call_type,
                        prompt_characters,
                        completion_characters,
                    );
                }
                if provider == Some(LlmProviders::TOGETHER_AI)
                    && matches!(
                        crate::call_type::CallTypes::parse(call_type),
                        Some(crate::call_type::CallTypes::embedding)
                            | Some(crate::call_type::CallTypes::aembedding)
                    )
                {
                    return together_ai_cost_per_token(self, request, call_type);
                }
                if provider == Some(LlmProviders::AZURE_AI) {
                    return azure_ai_cost_per_token(self, request, request_model);
                }
                Ok(self.cost_per_token(request)?)
            }
            CostCall::Speech { prompt_characters } => {
                Ok(self.speech_cost(request, prompt_characters)?)
            }
            CostCall::Transcription { duration_seconds } => {
                Ok(self.transcription_cost(request, duration_seconds)?)
            }
            CostCall::Rerank { billed_units } => {
                let provider = request.provider.ok_or(CostError::MissingProvider)?;
                Ok(self.rerank_cost(request.model, provider, request.region, billed_units))
            }
            CostCall::VectorStoreSearch { api_type } => {
                let provider = request.provider.ok_or(CostError::MissingProvider)?;
                Ok(self.vector_store_search_cost(provider, api_type))
            }
            CostCall::Search {
                number_of_queries,
                optional_params,
            } => Ok(self.search_provider_cost_per_query(
                request.model,
                request.provider,
                number_of_queries.filter(|count| *count > 0).unwrap_or(1),
                optional_params,
            )?),
            CostCall::Ocr {
                response,
                deployment_info,
            } => {
                let published = self
                    .select_model_key(request.model, request.provider, request.region)
                    .and_then(|key| self.entries.get(key));
                Ok(ocr_cost(response, deployment_info, published)?)
            }
            CostCall::Batch { deployment_info } => {
                let published = self
                    .select_model_key(request.model, request.provider, request.region)
                    .and_then(|key| self.entries.get(key));
                let has_deployment_rates = deployment_info.is_some_and(|info| {
                    [
                        "input_cost_per_token_batches",
                        "input_cost_per_token",
                        "output_cost_per_token_batches",
                        "output_cost_per_token",
                    ]
                    .into_iter()
                    .any(|key| info.get(key).is_some_and(|value| !value.is_null()))
                });
                let info = if has_deployment_rates {
                    deployment_info
                } else {
                    published.or(deployment_info)
                };
                let Some(info) = info else {
                    return Ok((0.0, 0.0));
                };
                let cost = batch_cost_from_model_info(
                    info,
                    request.usage,
                    request.provider,
                    request.data_residency,
                )?;
                Ok((cost.prompt, cost.completion))
            }
        }
    }

    pub fn speech_cost(
        &self,
        request: ModelCostRequest<'_>,
        prompt_characters: Option<f64>,
    ) -> Result<(f64, f64), CostError> {
        let key = self
            .select_model_key(request.model, request.provider, request.region)
            .ok_or(CostError::ModelNotFound)?;
        let model_info = &self.entries[key];
        if matches!(request.provider, Some("vertex_ai" | "vertex_ai_beta"))
            && let Some(cost) = lyria_generation_cost(model_info)
        {
            return Ok((0.0, cost));
        }
        match select_cost_metric_for_model(model_info)? {
            SpeechCostMetric::PerCharacter => {
                let characters = prompt_characters.ok_or(CostError::MissingPromptCharacters)?;
                let (prompt, completion) =
                    generic_cost_per_character(model_info, characters, 0.0, None, Some(0.0));
                Ok((
                    prompt.ok_or(CostError::MissingInputCharacterRate)?,
                    completion.unwrap_or(0.0),
                ))
            }
            SpeechCostMetric::PerToken => Ok(self.cost_per_token(request)?),
        }
    }

    pub fn transcription_cost(
        &self,
        request: ModelCostRequest<'_>,
        duration_seconds: f64,
    ) -> Result<(f64, f64), CostError> {
        let key = self
            .select_model_key(request.model, request.provider, request.region)
            .ok_or(CostError::ModelNotFound)?;
        let model_info = &self.entries[key];
        if transcription_usage_has_token_details(request.usage) {
            return Ok(calculate_generic_cost_from_model_info_with_region(
                request.usage,
                model_info,
                request.service_tier,
                false,
                request.data_residency,
                request.vertex_location,
                request.at,
            ));
        }
        Ok(cost_per_second(model_info, duration_seconds))
    }

    pub fn handle_realtime_transcription_cost_calculation(
        &self,
        results: &[Value],
        provider: &str,
        requested_model: &str,
    ) -> f64 {
        let model = get_transcription_model_name_from_results(results).unwrap_or(requested_model);
        let model_info = self
            .select_model_key(model, Some(provider), None)
            .and_then(|key| self.entries.get(key));
        results
            .iter()
            .filter(|event| {
                event.get("type").and_then(Value::as_str)
                    == Some("conversation.item.input_audio_transcription.completed")
            })
            .map(|event| transcription_usage_cost(&event["usage"], model_info))
            .sum()
    }

    pub fn handle_realtime_stream_cost_calculation(
        &self,
        results: &[Value],
        combined_usage: &ChatUsage,
        provider: &str,
        requested_model: &str,
        data_residency: Option<&str>,
        at: Timestamp,
    ) -> f64 {
        let models = results
            .iter()
            .filter(|event| event.get("type").and_then(Value::as_str) == Some("session.created"))
            .filter_map(|event| event.pointer("/session/model").and_then(Value::as_str))
            .chain(std::iter::once(requested_model));
        let token_cost = models
            .filter_map(|model| {
                let key = self.select_model_key(model, Some(provider), None)?;
                let cost = calculate_generic_cost_from_model_info_with_region(
                    combined_usage,
                    &self.entries[key],
                    None,
                    false,
                    data_residency,
                    None,
                    at,
                );
                let declares_pricing = [
                    self.entries.get(model),
                    self.entries.get(&format!("{provider}/{model}")),
                ]
                .into_iter()
                .flatten()
                .any(|entry| {
                    entry.as_object().is_some_and(|fields| {
                        fields
                            .iter()
                            .any(|(field, value)| field.contains("cost_per") && !value.is_null())
                    })
                });
                (cost.0 + cost.1 > 0.0 || declares_pricing).then_some(cost)
            })
            .next()
            .unwrap_or((0.0, 0.0));
        token_cost.0
            + token_cost.1
            + self.handle_realtime_transcription_cost_calculation(
                results,
                provider,
                requested_model,
            )
    }

    pub fn responses_ws_token_cost_by_tier(
        &self,
        results: &[Value],
        model: &str,
        provider: Option<&str>,
        region: Option<&str>,
        data_residency: Option<&str>,
        at: Timestamp,
    ) -> Result<f64, CostError> {
        partition_results_by_service_tier(results)
            .into_iter()
            .map(|(service_tier, events)| {
                let usage = combine_usage_objects(
                    events
                        .into_iter()
                        .map(event_usage)
                        .collect::<Result<Vec<_>, _>>()?,
                )?;
                let (prompt, completion) = self.cost_per_token(ModelCostRequest {
                    model,
                    provider,
                    region,
                    usage: &usage,
                    service_tier,
                    data_residency,
                    vertex_location: None,
                    at,
                    response_time_ms: None,
                })?;
                Ok(prompt + completion)
            })
            .sum()
    }

    pub fn cost_per_token(&self, request: ModelCostRequest<'_>) -> Result<(f64, f64), CostError> {
        if let Some(cost) = self
            .select_model_key(request.model, request.provider, request.region)
            .and_then(|key| self.entries.get(key))
            .and_then(|info| per_second_pricing_cost(info, request.response_time_ms))
        {
            return Ok(cost);
        }
        let provider = request.provider.and_then(LlmProviders::parse);
        if provider == Some(LlmProviders::AZURE_AI) {
            return azure_ai_cost_per_token(self, request, None);
        }
        if provider == Some(LlmProviders::DATABRICKS) {
            return databricks_cost_per_token(self, request);
        }
        if provider == Some(LlmProviders::LEMONADE) {
            return Ok(lemonade_cost_per_token(self, request));
        }
        if provider == Some(LlmProviders::PERPLEXITY)
            && let Some(cost) = request.usage.cost
        {
            return Ok((0.0, cost));
        }
        if provider == Some(LlmProviders::XAI)
            && let Some(cost) = xai_reported_cost(request.usage)
        {
            return Ok((0.0, cost));
        }
        let needs_together_fallback = (provider == Some(LlmProviders::TOGETHER_AI)
            || request.model.contains("togethercomputer")
            || request.model.contains("together_ai"))
            && !has_together_registry_pricing(request.model, &self.entries);
        let together_fallback = needs_together_fallback.then(|| {
            get_model_params_and_category(
                request.model,
                "completion",
                TogetherThresholds::default(),
            )
        });
        let key = match together_fallback.as_deref() {
            Some(category) => self.select_model_key(category, None, request.region),
            None => self.select_model_key(request.model, request.provider, request.region),
        }
        .or_else(|| {
            (request.provider == Some("fireworks_ai"))
                .then(|| get_base_model_for_pricing(request.model, FireworksThresholds::default()))
                .and_then(|category| {
                    self.select_model_key(category, request.provider, request.region)
                })
        })
        .ok_or(CostError::ModelNotFound)?;
        let model_info = apply_provider_cache_read_default(&self.entries[key], request.provider);
        if let Some(cost) = per_second_pricing_cost(&model_info, request.response_time_ms) {
            return Ok(cost);
        }
        if provider == Some(LlmProviders::AZURE)
            && let Some(cost) = output_per_second_cost(&model_info, request.response_time_ms)
        {
            return Ok(cost);
        }
        match provider {
            Some(LlmProviders::PERPLEXITY) => {
                return Ok(perplexity_cost_per_token(
                    request.usage,
                    &model_info,
                    request.at,
                ));
            }
            Some(LlmProviders::XAI) => {
                return Ok(xai_cost_per_token(request.usage, &model_info, request.at));
            }
            Some(
                LlmProviders::DASHSCOPE | LlmProviders::QWENCLOUD | LlmProviders::QWEN_AI_PLATFORM,
            ) => {
                return Ok(dashscope_cost_per_token(
                    request.usage,
                    &model_info,
                    request.at,
                ));
            }
            Some(LlmProviders::FIREWORKS_AI) => {
                return Ok(fireworks_cost_per_token(
                    request.usage,
                    &self.entries[key],
                    request.at,
                ));
            }
            Some(LlmProviders::VERTEX_AI) => {
                return Ok(vertex_cost_per_token(
                    request.usage,
                    &model_info,
                    request.service_tier,
                    request.vertex_location,
                    request.at,
                ));
            }
            _ => {}
        }
        let dispatches_before_the_gate = matches!(
            provider,
            Some(
                LlmProviders::VERTEX_AI
                    | LlmProviders::ANTHROPIC
                    | LlmProviders::BEDROCK
                    | LlmProviders::OPENAI
                    | LlmProviders::DATABRICKS
                    | LlmProviders::FIREWORKS_AI
                    | LlmProviders::AZURE
                    | LlmProviders::GEMINI
                    | LlmProviders::DEEPSEEK
                    | LlmProviders::TENCENT
                    | LlmProviders::PERPLEXITY
                    | LlmProviders::XAI
                    | LlmProviders::LEMONADE
                    | LlmProviders::DASHSCOPE
                    | LlmProviders::QWENCLOUD
                    | LlmProviders::QWEN_AI_PLATFORM
                    | LlmProviders::AZURE_AI
            )
        );
        if !dispatches_before_the_gate && !has_token_or_tiered_pricing(&model_info) {
            return Ok((0.0, 0.0));
        }
        let cost = calculate_generic_cost_from_model_info_with_region(
            request.usage,
            &model_info,
            request.service_tier,
            uses_inclusive_token_thresholds(request.provider),
            request.data_residency,
            request.vertex_location,
            request.at,
        );
        let speed = if provider == Some(LlmProviders::ANTHROPIC) {
            fast_speed_multiplier(&model_info, request.usage)
                * get_provider_specific_geo_multiplier(
                    &model_info,
                    request
                        .usage
                        .extra
                        .get("inference_geo")
                        .and_then(Value::as_str),
                )
        } else {
            1.0
        };
        Ok((cost.0 * speed, cost.1 * speed))
    }

    pub fn get_billed_token_rates(
        &self,
        request: ModelCostRequest<'_>,
        custom_cost_per_token: Option<CustomTokenRates>,
    ) -> Option<BilledTokenRates> {
        let model_info = self
            .select_model_key(request.model, request.provider, request.region)
            .and_then(|key| self.entries.get(key));
        calculate_billed_token_rates(BilledRatesRequest {
            model_info,
            usage: request.usage,
            provider: request.provider,
            service_tier: request.service_tier,
            data_residency: request.data_residency,
            vertex_location: request.vertex_location,
            at: request.at,
            custom_cost_per_token,
        })
    }

    pub fn get_token_type_cost_breakdown(
        &self,
        request: ModelCostRequest<'_>,
        custom_cost_per_token: Option<CustomTokenRates>,
    ) -> TokenTypeCostBreakdown {
        let model_info = self
            .select_model_key(request.model, request.provider, request.region)
            .and_then(|key| self.entries.get(key));
        calculate_token_type_cost_breakdown(BilledRatesRequest {
            model_info,
            usage: request.usage,
            provider: request.provider,
            service_tier: request.service_tier,
            data_residency: request.data_residency,
            vertex_location: request.vertex_location,
            at: request.at,
            custom_cost_per_token,
        })
    }

    pub fn calculate_prompt_caching_savings(&self, request: ModelCostRequest<'_>) -> Option<f64> {
        let model_info = self
            .select_model_key(request.model, request.provider, request.region)
            .and_then(|key| self.entries.get(key))?;
        Some(calculate_prompt_caching_savings(
            PromptCachingSavingsRequest {
                model_info,
                usage: request.usage,
                provider: request.provider,
                service_tier: request.service_tier,
                data_residency: request.data_residency,
                vertex_location: request.vertex_location,
                at: request.at,
            },
        ))
    }

    pub fn calculate_image_response_cost_from_usage(
        &self,
        model: &str,
        provider: Option<&str>,
        region: Option<&str>,
        image_response: &Value,
        at: Timestamp,
    ) -> Option<f64> {
        let model_info = self
            .select_model_key(model, provider, region)
            .and_then(|key| self.entries.get(key))?;
        calculate_image_response_cost_from_usage(image_response, model_info, provider, at)
    }

    pub fn google_image_generation_cost(
        &self,
        model: &str,
        provider: &str,
        image_response: &Value,
        supplied_model_info: Option<&Value>,
        at: Timestamp,
    ) -> Result<f64, CostError> {
        let shared = self
            .select_model_key(model, Some(provider), None)
            .and_then(|key| self.entries.get(key));
        let model_info = resolve_image_model_info(shared, supplied_model_info)
            .ok_or(CostError::ModelNotFound)?;
        match provider {
            "gemini" => Ok(gemini_image_generation_cost(
                image_response,
                &model_info,
                at,
            )),
            "vertex_ai" => Ok(vertex_image_generation_cost(
                image_response,
                &model_info,
                at,
            )),
            _ => Err(CostError::ModelNotFound),
        }
    }

    pub fn google_image_edit_cost(
        &self,
        model: &str,
        provider: &str,
        image_response: &Value,
        supplied_model_info: Option<&Value>,
        at: Timestamp,
    ) -> Result<f64, CostError> {
        let shared = self
            .select_model_key(model, Some(provider), None)
            .and_then(|key| self.entries.get(key));
        match provider {
            "gemini" => {
                let info = resolve_image_model_info(shared, supplied_model_info)
                    .ok_or(CostError::ModelNotFound)?;
                Ok(gemini_image_edit_cost(image_response, &info, at))
            }
            "vertex_ai" => shared
                .map(|info| vertex_image_edit_cost(image_response, info))
                .ok_or(CostError::ModelNotFound),
            _ => Err(CostError::ModelNotFound),
        }
    }

    pub fn azure_ai_image_generation_cost(
        &self,
        request: AzureAiImageCatalogRequest<'_>,
    ) -> Result<f64, CostError> {
        let shared = self
            .select_model_key(request.model, Some("azure_ai"), None)
            .and_then(|key| self.entries.get(key));
        let model_info = resolve_image_model_info(shared, request.supplied_model_info)
            .ok_or(CostError::ModelNotFound)?;
        let shared_pricing = model_info
            .get("key")
            .and_then(Value::as_str)
            .and_then(|key| self.entries.get(key))
            .or(shared);
        azure_ai_image_cost_calculator(AzureAiImageRequest {
            image_response: request.image_response,
            model_info: &model_info,
            supplied_model_info: request.supplied_model_info,
            shared_pricing,
            size: request.size,
            n: request.n,
            optional_params: request.optional_params,
            at: request.at,
        })
    }

    pub fn flat_image_generation_cost(
        &self,
        model: &str,
        provider: &str,
        image_response: &Value,
        supplied_model_info: Option<&Value>,
    ) -> Result<f64, CostError> {
        let shared = self
            .select_model_key(model, Some(provider), None)
            .and_then(|key| self.entries.get(key));
        let model_info = resolve_image_model_info(shared, supplied_model_info)
            .ok_or(CostError::ModelNotFound)?;
        Ok(flat_image_cost(image_response, &model_info))
    }

    pub fn openai_image_generation_cost(
        &self,
        model: &str,
        provider: &str,
        image_response: &Value,
        supplied_model_info: Option<&Value>,
        at: Timestamp,
    ) -> Result<f64, CostError> {
        let shared = self
            .select_model_key(model, Some(provider), None)
            .and_then(|key| self.entries.get(key));
        let model_info = resolve_image_model_info(shared, supplied_model_info)
            .ok_or(CostError::ModelNotFound)?;
        openai_image_cost_calculator(image_response, &model_info, provider, at)
    }

    pub fn fal_ai_image_generation_cost(
        &self,
        model: &str,
        image_response: &Value,
        optional_params: &Value,
        deployment_prices: Option<&Value>,
    ) -> Result<f64, CostError> {
        fal_ai_image_cost_calculator(
            model,
            image_response,
            optional_params,
            deployment_prices,
            &self.entries,
        )
        .ok_or(CostError::ModelNotFound)
    }

    pub fn fal_ai_passthrough_cost(&self, model: &str, request_body: &Value) -> Option<f64> {
        let normalized = model.strip_prefix("fal_ai/").unwrap_or(model);
        calculate_fal_ai_passthrough_cost(
            self.entries.get(&format!("fal_ai/{normalized}")),
            request_body,
        )
    }

    pub fn bedrock_image_generation_cost(
        &self,
        model: &str,
        image_response: &Value,
        size: Option<&str>,
        optional_params: &Value,
    ) -> Result<f64, CostError> {
        bedrock_image_cost_calculator(model, image_response, size, optional_params, &self.entries)
            .ok_or(CostError::ModelNotFound)
    }

    pub fn video_generation_cost(
        &self,
        model: &str,
        provider: Option<&str>,
        deployment_info: Option<&Value>,
        duration_seconds: f64,
        video_resolution: Option<&str>,
    ) -> Result<f64, CostError> {
        let model_info = deployment_info
            .or_else(|| {
                self.select_model_key(model, provider, None)
                    .and_then(|key| self.entries.get(key))
            })
            .ok_or(CostError::ModelNotFound)?;
        calculate_video_generation_cost(model_info, duration_seconds, video_resolution)
    }

    pub fn default_image_cost_calculator(
        &self,
        request: DefaultImageCostRequest<'_>,
    ) -> Result<f64, CostError> {
        let raw_size = request.size.unwrap_or("1024-x-1024");
        let size = if raw_size.contains("-x-") {
            raw_size.to_owned()
        } else {
            raw_size.replace('x', "-x-")
        };
        let (height, width) = size.split_once("-x-").ok_or(CostError::InvalidQuantity)?;
        let height = height
            .parse::<u32>()
            .map_err(|_| CostError::InvalidQuantity)?;
        let width = width
            .parse::<u32>()
            .map_err(|_| CostError::InvalidQuantity)?;
        let provider_prefix = request.provider.map(|provider| format!("{provider}/"));
        let without_provider = provider_prefix
            .as_deref()
            .and_then(|prefix| request.model.strip_prefix(prefix));
        let base = match (request.provider, without_provider) {
            (Some(provider), Some(model)) => format!("{provider}/{size}/{model}"),
            _ => format!("{size}/{}", request.model),
        };
        let model_tail = request.model.rsplit('/').next().unwrap_or(request.model);
        let without_prefix = format!("{size}/{model_tail}");
        let candidates = [
            request.quality.map(|quality| format!("{quality}/{base}")),
            match (request.provider, request.quality) {
                (Some(provider), Some(quality)) => Some(format!(
                    "{provider}/{quality}/{size}/{}",
                    without_provider.unwrap_or(request.model)
                )),
                _ => None,
            },
            Some(base.clone()),
            Some(format!("high/{base}")),
            request
                .quality
                .map(|quality| format!("{quality}/{without_prefix}")),
            Some(without_prefix),
            Some(request.model.to_owned()),
            without_provider.map(str::to_owned),
        ];
        let shared = candidates
            .iter()
            .flatten()
            .find_map(|candidate| self.entries.get(candidate));
        if shared.is_none() && request.supplied_model_info.is_none() {
            return Err(CostError::ModelNotFound);
        }
        let rate = |info: &Value, key: &str| {
            crate::generic_input::get_cost_per_unit(info, key, None)
                .map_or(crate::pricing::Rate::Missing, crate::pricing::Rate::Value)
        };
        let tables = [request.supplied_model_info, shared]
            .into_iter()
            .flatten()
            .map(|info| ImageRates {
                input_per_image: rate(info, "input_cost_per_image"),
                output_per_image: rate(info, "output_cost_per_image"),
                input_per_pixel: rate(info, "input_cost_per_pixel"),
            })
            .collect::<Vec<_>>();
        Ok(calculate_image(
            &tables,
            ImageUsage {
                count: request.n.unwrap_or(1),
                width,
                height,
            },
        )?
        .total)
    }

    pub fn rerank_cost(
        &self,
        model: &str,
        provider: &str,
        region: Option<&str>,
        billed_units: Option<&Value>,
    ) -> (f64, f64) {
        let model_info = self
            .select_model_key(model, Some(provider), region)
            .and_then(|key| self.entries.get(key));
        rerank_cost(provider, model_info, billed_units)
    }

    pub fn vector_store_search_cost(&self, provider: &str, api_type: Option<&str>) -> (f64, f64) {
        vector_store_search_cost(provider, api_type, self.entries.get("vertex_ai/search_api"))
    }

    pub fn search_provider_cost_per_query(
        &self,
        model: &str,
        provider: Option<&str>,
        number_of_queries: u64,
        optional_params: &Value,
    ) -> Result<(f64, f64), CostError> {
        if provider == Some("parallel_ai") {
            let pricing_model = match effective_mode(optional_params) {
                "fast" => "parallel_ai/search-fast",
                "turbo" => "parallel_ai/search-turbo",
                _ => "parallel_ai/search",
            };
            let model_info = self
                .entries
                .get(pricing_model)
                .ok_or(CostError::ModelNotFound)?;
            let request_cost = model_info
                .get("input_cost_per_query")
                .and_then(|value| match value {
                    Value::Number(value) => value.as_f64(),
                    Value::String(value) => value.parse().ok(),
                    _ => None,
                })
                .unwrap_or(0.0);
            return Ok((
                parallel_ai_search_cost(
                    optional_params,
                    provider_usage(optional_params),
                    ParallelAiPricing {
                        request_cost,
                        default_results: 10,
                        additional_result_cost: 0.001,
                    },
                ),
                0.0,
            ));
        }
        let model_info = self
            .select_model_key(model, provider, None)
            .and_then(|key| self.entries.get(key))
            .ok_or(CostError::ModelNotFound)?;
        Ok(search_provider_cost_per_query(
            model_info,
            number_of_queries,
            optional_params,
        ))
    }

    pub fn built_in_tool_cost(
        &self,
        model: &str,
        provider: Option<&str>,
        region: Option<&str>,
        request: BuiltInToolCostRequest<'_>,
    ) -> f64 {
        let model_info = self
            .select_model_key(model, provider, region)
            .and_then(|key| self.entries.get(key));
        get_cost_for_built_in_tools(request, model_info)
    }

    pub fn completion_cost(
        &self,
        request: CompletionCostRequest<'_>,
    ) -> Result<CompletionCost, CostError> {
        let (prompt, output) = self.cost_per_token(request.token)?;
        let built_in_tools = match request.built_in_tools {
            BuiltInToolCharge::Provided(cost) => cost,
            BuiltInToolCharge::FromResponse(tool_request) => self.built_in_tool_cost(
                request.token.model,
                request.token.provider,
                request.token.region,
                tool_request,
            ),
        };
        Ok(completion_cost(
            prompt,
            output,
            built_in_tools,
            request.additional_costs,
            request.token.provider,
            request.discount_config,
            request.margin_config,
        ))
    }

    pub fn response_cost_calculator(
        &self,
        request: ResponseCostRequest<'_>,
    ) -> Result<f64, CostError> {
        if request.cache_hit {
            return Ok(0.0);
        }
        if let Some(reported) = get_response_cost_from_hidden_params(request.hidden_params)? {
            return Ok(reported);
        }
        Ok(self.completion_cost(request.completion)?.total)
    }
}
