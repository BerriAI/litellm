use std::collections::HashMap;

use jiff::Timestamp;
use serde_json::Value;

use crate::anthropic_cost::fast_speed_multiplier;
use crate::azure_ai_cost::{
    calculate_azure_model_router_flat_cost, is_azure_model_router, router_fee_entry_name,
    router_fee_name,
};
use crate::azure_ai_image_cost::{
    AzureAiImageRequest, cost_calculator as azure_ai_image_cost_calculator,
};
use crate::azure_cost::output_per_second_cost;
use crate::billed_token_rates::{
    BilledRatesRequest, BilledTokenRates, TokenTypeCostBreakdown,
    get_billed_token_rates as calculate_billed_token_rates,
    get_token_type_cost_breakdown as calculate_token_type_cost_breakdown,
};
use crate::completion_cost::{
    CompletionCost, ResponseCostError, completion_cost, get_response_cost_from_hidden_params,
};
use crate::custom_pricing::CustomTokenRates;
use crate::dashscope_cost::cost_per_token as dashscope_cost_per_token;
use crate::databricks_cost::registry_key as databricks_registry_key;
use crate::fireworks_cost::{
    FireworksThresholds, cost_per_token as fireworks_cost_per_token, get_base_model_for_pricing,
};
use crate::generic_cost::calculate_generic_cost_from_model_info_with_region;
use crate::image_response_cost::{
    calculate_image_response_cost_from_usage, flat_image_cost, gemini_image_edit_cost,
    gemini_image_generation_cost, resolve_image_model_info, vertex_image_edit_cost,
    vertex_image_generation_cost,
};
use crate::non_token::Error as NonTokenError;
use crate::per_second::per_second_pricing_cost;
use crate::perplexity_cost::cost_per_token as perplexity_cost_per_token;
use crate::prompt_caching_savings::{
    PromptCachingSavingsRequest, calculate_prompt_caching_savings,
};
use crate::provider_cache::apply_provider_cache_read_default;
use crate::realtime_cost::{
    combine_usage_objects, event_usage, get_transcription_model_name_from_results,
    partition_results_by_service_tier, transcription_usage_cost,
};
use crate::responses_usage::{ChatUsage, UsageError};
use crate::retrieval_cost::{rerank_cost, vector_store_search_cost};
use crate::search_cost::{
    ParallelAiPricing, effective_mode, parallel_ai_search_cost, provider_usage,
    search_provider_cost_per_query,
};
use crate::speech_cost::{
    SpeechCostError, SpeechCostMetric, cost_per_second, generic_cost_per_character,
    lyria_generation_cost, select_cost_metric_for_model, transcription_usage_has_token_details,
};
use crate::tool_cost_dispatch::{BuiltInToolCostRequest, get_cost_for_built_in_tools};
use crate::vertex_cost::{
    CostRoute, cost_per_character as vertex_cost_per_character,
    cost_per_token as vertex_cost_per_token, cost_router as vertex_cost_router,
};
use crate::xai_cost::{cost_per_token as xai_cost_per_token, reported_cost as xai_reported_cost};
use crate::{Cost, Pricing, PricingError, Rates, Request, calculate};

#[derive(Clone, Debug, Default)]
pub struct CostCatalog {
    entries: HashMap<String, Rates>,
}

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CatalogError {
    ModelNotFound,
    Pricing(PricingError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CatalogResponseError {
    Catalog(CatalogError),
    ProviderCost(ResponseCostError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RealtimeCostError {
    Catalog(CatalogError),
    Usage(UsageError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CatalogSpeechError {
    Catalog(CatalogError),
    Speech(SpeechCostError),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CatalogImageError {
    Catalog(CatalogError),
    Pricing(NonTokenError),
}

impl From<CatalogError> for CatalogImageError {
    fn from(value: CatalogError) -> Self {
        Self::Catalog(value)
    }
}

impl From<NonTokenError> for CatalogImageError {
    fn from(value: NonTokenError) -> Self {
        Self::Pricing(value)
    }
}

impl From<CatalogError> for CatalogSpeechError {
    fn from(value: CatalogError) -> Self {
        Self::Catalog(value)
    }
}

impl From<SpeechCostError> for CatalogSpeechError {
    fn from(value: SpeechCostError) -> Self {
        Self::Speech(value)
    }
}

impl From<CatalogError> for RealtimeCostError {
    fn from(value: CatalogError) -> Self {
        Self::Catalog(value)
    }
}

impl From<UsageError> for RealtimeCostError {
    fn from(value: UsageError) -> Self {
        Self::Usage(value)
    }
}

impl From<PricingError> for CatalogError {
    fn from(value: PricingError) -> Self {
        Self::Pricing(value)
    }
}

impl From<CatalogError> for CatalogResponseError {
    fn from(value: CatalogError) -> Self {
        Self::Catalog(value)
    }
}

impl From<ResponseCostError> for CatalogResponseError {
    fn from(value: ResponseCostError) -> Self {
        Self::ProviderCost(value)
    }
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

impl CostCatalog {
    pub fn new(entries: HashMap<String, Rates>) -> Self {
        Self { entries }
    }

    pub fn select_model_key<'a>(
        &'a self,
        model: &str,
        provider: Option<&str>,
        region: Option<&str>,
    ) -> Option<&'a str> {
        select_model_key(&self.entries, model, provider, region)
    }

    pub fn cost_per_token(
        &self,
        model: &str,
        provider: Option<&str>,
        region: Option<&str>,
        request: &Request,
    ) -> Result<Cost, CatalogError> {
        let key = self
            .select_model_key(model, provider, region)
            .ok_or(CatalogError::ModelNotFound)?;
        let pricing = Pricing {
            standard: self.entries[key],
            tiers: &[],
            thresholds: &[],
            off_peak: None,
        };
        Ok(calculate(&pricing, request)?)
    }
}

impl ModelInfoCatalog {
    pub fn new(entries: HashMap<String, Value>) -> Self {
        Self { entries }
    }

    pub fn select_model_key<'a>(
        &'a self,
        model: &str,
        provider: Option<&str>,
        region: Option<&str>,
    ) -> Option<&'a str> {
        select_model_key(&self.entries, model, provider, region)
    }

    pub fn vertex_cost(
        &self,
        request: ModelCostRequest<'_>,
        call_type: &str,
        prompt_characters: Option<f64>,
        completion_characters: Option<f64>,
    ) -> Result<(f64, f64), CatalogError> {
        if vertex_cost_router(request.model, request.provider.unwrap_or(""), call_type)
            == CostRoute::PerToken
        {
            return self.cost_per_token(request);
        }
        let key = self
            .select_model_key(request.model, request.provider, request.region)
            .ok_or(CatalogError::ModelNotFound)?;
        Ok(vertex_cost_per_character(
            request.model,
            request.usage,
            &self.entries[key],
            (prompt_characters, completion_characters),
            request.service_tier,
            request.vertex_location,
            request.at,
        ))
    }

    pub fn databricks_cost_per_token(
        &self,
        request: ModelCostRequest<'_>,
    ) -> Result<(f64, f64), CatalogError> {
        if let Some(cost) = self
            .select_model_key(request.model, Some("databricks"), request.region)
            .and_then(|key| self.entries.get(key))
            .and_then(|info| per_second_pricing_cost(info, request.response_time_ms))
        {
            return Ok(cost);
        }
        let key = self
            .select_model_key(
                databricks_registry_key(request.model),
                Some("databricks"),
                request.region,
            )
            .ok_or(CatalogError::ModelNotFound)?;
        Ok(calculate_generic_cost_from_model_info_with_region(
            request.usage,
            &self.entries[key],
            None,
            false,
            None,
            None,
            request.at,
        ))
    }

    pub fn lemonade_cost_per_token(&self, _request: ModelCostRequest<'_>) -> (f64, f64) {
        (0.0, 0.0)
    }

    pub fn azure_ai_cost_per_token(
        &self,
        request: ModelCostRequest<'_>,
        request_model: Option<&str>,
    ) -> Result<(f64, f64), CatalogError> {
        let model_info = self
            .select_model_key(request.model, Some("azure_ai"), request.region)
            .and_then(|key| self.entries.get(key));
        if let Some(cost) =
            model_info.and_then(|info| per_second_pricing_cost(info, request.response_time_ms))
        {
            return Ok(cost);
        }
        let (prompt, completion) = match model_info {
            Some(info) => calculate_generic_cost_from_model_info_with_region(
                request.usage,
                info,
                request.service_tier,
                false,
                request.data_residency,
                request.vertex_location,
                request.at,
            ),
            None if is_azure_model_router(request.model) => (0.0, 0.0),
            None => return Err(CatalogError::ModelNotFound),
        };
        let Some(fee_name) = router_fee_name(request.model, request_model) else {
            return Ok((prompt, completion));
        };
        let fee_entry = router_fee_entry_name(fee_name);
        let fee_key = self
            .select_model_key(fee_entry, Some("azure_ai"), None)
            .ok_or(CatalogError::ModelNotFound)?;
        let fee = calculate_azure_model_router_flat_cost(
            fee_name,
            request.usage.prompt_tokens,
            &self.entries[fee_key],
        );
        Ok((prompt + fee, completion))
    }

    pub fn speech_cost(
        &self,
        request: ModelCostRequest<'_>,
        prompt_characters: Option<f64>,
    ) -> Result<(f64, f64), CatalogSpeechError> {
        let key = self
            .select_model_key(request.model, request.provider, request.region)
            .ok_or(CatalogError::ModelNotFound)?;
        let model_info = &self.entries[key];
        if matches!(request.provider, Some("vertex_ai" | "vertex_ai_beta"))
            && let Some(cost) = lyria_generation_cost(model_info)
        {
            return Ok((0.0, cost));
        }
        match select_cost_metric_for_model(model_info)? {
            SpeechCostMetric::PerCharacter => {
                let characters =
                    prompt_characters.ok_or(SpeechCostError::MissingPromptCharacters)?;
                let (prompt, completion) =
                    generic_cost_per_character(model_info, characters, 0.0, None, Some(0.0));
                Ok((
                    prompt.ok_or(SpeechCostError::MissingInputCharacterRate)?,
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
    ) -> Result<(f64, f64), CatalogError> {
        let key = self
            .select_model_key(request.model, request.provider, request.region)
            .ok_or(CatalogError::ModelNotFound)?;
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
    ) -> Result<f64, RealtimeCostError> {
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

    pub fn cost_per_token(
        &self,
        request: ModelCostRequest<'_>,
    ) -> Result<(f64, f64), CatalogError> {
        if request.provider == Some("azure_ai") {
            return self.azure_ai_cost_per_token(request, None);
        }
        if request.provider == Some("databricks") {
            return self.databricks_cost_per_token(request);
        }
        if request.provider == Some("lemonade") {
            let per_second = self
                .select_model_key(request.model, Some("lemonade"), request.region)
                .and_then(|key| self.entries.get(key))
                .and_then(|info| per_second_pricing_cost(info, request.response_time_ms));
            return Ok(per_second.unwrap_or_else(|| self.lemonade_cost_per_token(request)));
        }
        if request.provider == Some("perplexity")
            && let Some(cost) = request.usage.cost
        {
            return Ok((0.0, cost));
        }
        if request.provider == Some("xai")
            && let Some(cost) = xai_reported_cost(request.usage)
        {
            return Ok((0.0, cost));
        }
        let key = self
            .select_model_key(request.model, request.provider, request.region)
            .or_else(|| {
                (request.provider == Some("fireworks_ai"))
                    .then(|| {
                        get_base_model_for_pricing(request.model, FireworksThresholds::default())
                    })
                    .and_then(|category| {
                        self.select_model_key(category, request.provider, request.region)
                    })
            })
            .ok_or(CatalogError::ModelNotFound)?;
        let model_info = apply_provider_cache_read_default(&self.entries[key], request.provider);
        if let Some(cost) = per_second_pricing_cost(&model_info, request.response_time_ms) {
            return Ok(cost);
        }
        if request.provider == Some("azure")
            && let Some(cost) = output_per_second_cost(&model_info, request.response_time_ms)
        {
            return Ok(cost);
        }
        if request.provider == Some("perplexity") {
            return Ok(perplexity_cost_per_token(
                request.usage,
                &model_info,
                request.at,
            ));
        }
        if request.provider == Some("xai") {
            return Ok(xai_cost_per_token(request.usage, &model_info, request.at));
        }
        if matches!(
            request.provider,
            Some("dashscope" | "qwencloud" | "qwen_ai_platform")
        ) {
            return Ok(dashscope_cost_per_token(
                request.usage,
                &model_info,
                request.at,
            ));
        }
        if request.provider == Some("fireworks_ai") {
            return Ok(fireworks_cost_per_token(
                request.usage,
                &self.entries[key],
                request.at,
            ));
        }
        if request.provider == Some("vertex_ai") {
            return Ok(vertex_cost_per_token(
                request.usage,
                &model_info,
                request.service_tier,
                request.vertex_location,
                request.at,
            ));
        }
        let cost = calculate_generic_cost_from_model_info_with_region(
            request.usage,
            &model_info,
            request.service_tier,
            request.provider == Some("xai"),
            request.data_residency,
            request.vertex_location,
            request.at,
        );
        let speed = if request.provider == Some("anthropic") {
            fast_speed_multiplier(&model_info, request.usage)
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
    ) -> Result<f64, CatalogError> {
        let shared = self
            .select_model_key(model, Some(provider), None)
            .and_then(|key| self.entries.get(key));
        let model_info = resolve_image_model_info(shared, supplied_model_info)
            .ok_or(CatalogError::ModelNotFound)?;
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
            _ => Err(CatalogError::ModelNotFound),
        }
    }

    pub fn google_image_edit_cost(
        &self,
        model: &str,
        provider: &str,
        image_response: &Value,
        supplied_model_info: Option<&Value>,
        at: Timestamp,
    ) -> Result<f64, CatalogError> {
        let shared = self
            .select_model_key(model, Some(provider), None)
            .and_then(|key| self.entries.get(key));
        match provider {
            "gemini" => {
                let info = resolve_image_model_info(shared, supplied_model_info)
                    .ok_or(CatalogError::ModelNotFound)?;
                Ok(gemini_image_edit_cost(image_response, &info, at))
            }
            "vertex_ai" => shared
                .map(|info| vertex_image_edit_cost(image_response, info))
                .ok_or(CatalogError::ModelNotFound),
            _ => Err(CatalogError::ModelNotFound),
        }
    }

    pub fn azure_ai_image_generation_cost(
        &self,
        request: AzureAiImageCatalogRequest<'_>,
    ) -> Result<f64, CatalogImageError> {
        let shared = self
            .select_model_key(request.model, Some("azure_ai"), None)
            .and_then(|key| self.entries.get(key));
        let model_info = resolve_image_model_info(shared, request.supplied_model_info)
            .ok_or(CatalogError::ModelNotFound)?;
        let shared_pricing = model_info
            .get("key")
            .and_then(Value::as_str)
            .and_then(|key| self.entries.get(key))
            .or(shared);
        Ok(azure_ai_image_cost_calculator(AzureAiImageRequest {
            image_response: request.image_response,
            model_info: &model_info,
            supplied_model_info: request.supplied_model_info,
            shared_pricing,
            size: request.size,
            n: request.n,
            optional_params: request.optional_params,
            at: request.at,
        })?)
    }

    pub fn flat_image_generation_cost(
        &self,
        model: &str,
        provider: &str,
        image_response: &Value,
        supplied_model_info: Option<&Value>,
    ) -> Result<f64, CatalogError> {
        let shared = self
            .select_model_key(model, Some(provider), None)
            .and_then(|key| self.entries.get(key));
        let model_info = resolve_image_model_info(shared, supplied_model_info)
            .ok_or(CatalogError::ModelNotFound)?;
        Ok(flat_image_cost(image_response, &model_info))
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
    ) -> Result<(f64, f64), CatalogError> {
        if provider == Some("parallel_ai") {
            let pricing_model = match effective_mode(optional_params) {
                "fast" => "parallel_ai/search-fast",
                "turbo" => "parallel_ai/search-turbo",
                _ => "parallel_ai/search",
            };
            let model_info = self
                .entries
                .get(pricing_model)
                .ok_or(CatalogError::ModelNotFound)?;
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
            .ok_or(CatalogError::ModelNotFound)?;
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
    ) -> Result<CompletionCost, CatalogError> {
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
    ) -> Result<f64, CatalogResponseError> {
        if request.cache_hit {
            return Ok(0.0);
        }
        if let Some(reported) = get_response_cost_from_hidden_params(request.hidden_params)? {
            return Ok(reported);
        }
        Ok(self.completion_cost(request.completion)?.total)
    }
}
