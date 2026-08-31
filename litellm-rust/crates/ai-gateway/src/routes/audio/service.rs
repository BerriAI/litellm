//! Audio service logic with full middleware.

use std::sync::Arc;
use std::time::Instant;

use base64::Engine;
use litellm_core::audio::types::{SpeechRequest, TranscriptionRequest};
use litellm_core::audio::{speech, transcription};
use litellm_core::auth::{HashedToken, KeyObject};
use litellm_core::cost_calculator::{self, CostRequest};
use litellm_core::persistence::CacheStore;
use litellm_core::spend_tracking::{EntityType, SpendUpdateItem};
use litellm_core::{CoreError, CoreResult};
use serde_json::{Map, Value};

use crate::integrations::custom_guardrail::{GuardrailContext, GuardrailRequest};
use crate::integrations::custom_logger::CallType;
use crate::state::AppState;

pub(crate) enum AudioResponseEnum {
    Binary { data: Vec<u8>, content_type: String },
    Json(Value),
}

pub async fn run_speech(
    state: &AppState,
    body: Value,
    _extra_headers: Option<Map<String, Value>>,
    key_object: &Arc<KeyObject>,
    hashed_token: &HashedToken,
) -> CoreResult<AudioResponseEnum> {
    let start = Instant::now();

    if !key_object.has_route_access(crate::constants::AUDIO_ROUTE_PATH_SPEECH) {
        return Err(CoreError::Auth(
            "API key does not have access to this route".to_string(),
        ));
    }

    let model = body
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .ok_or_else(|| CoreError::InvalidRequest("speech body requires a model".to_string()))?
        .to_string();

    let input = body
        .get("input")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|input| !input.is_empty())
        .ok_or_else(|| CoreError::InvalidRequest("speech body requires an input".to_string()))?
        .to_string();

    let voice = body
        .get("voice")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|voice| !voice.is_empty())
        .ok_or_else(|| CoreError::InvalidRequest("speech body requires a voice".to_string()))?
        .to_string();

    // Check global rate limit
    if !state.global_rate_limiter.check().await {
        return Err(CoreError::Auth("global rate limit exceeded".to_string()));
    }

    // Check model access
    if !key_object.has_model_access(&model) {
        return Err(CoreError::Auth(format!(
            "API key does not have access to model '{model}'"
        )));
    }

    // Check budget
    if !key_object.is_within_budget() {
        return Err(CoreError::Auth(format!(
            "API key has exceeded its budget limit of ${:.2}",
            key_object.max_budget.unwrap_or(0.0)
        )));
    }

    // Check team/org budget via Redis spend counters
    if let Some(ref redis) = state.redis {
        if let Some(ref team_id) = key_object.team_id {
            let mut key_buf = [0u8; 256];
            let key = spend_key(&mut key_buf, "team", team_id);
            match redis.incr_by_float(key, 0.0).await {
                Ok(team_spend) => {
                    let team_budget = state.config.team_budget;
                    if team_spend >= team_budget {
                        return Err(CoreError::Auth(format!(
                            "team '{team_id}' has exceeded its budget (${team_spend:.2} / ${team_budget:.2})"
                        )));
                    }
                }
                Err(e) => {
                    crate::hardening::log_degradation("redis", "team_budget_check", &e.to_string());
                }
            }
        }

        if let Some(ref org_id) = key_object.org_id {
            let mut key_buf = [0u8; 256];
            let key = spend_key(&mut key_buf, "org", org_id);
            match redis.incr_by_float(key, 0.0).await {
                Ok(org_spend) => {
                    let org_budget = state.config.org_budget;
                    if org_spend >= org_budget {
                        return Err(CoreError::Auth(format!(
                            "organization '{org_id}' has exceeded its budget (${org_spend:.2} / ${org_budget:.2})"
                        )));
                    }
                }
                Err(e) => {
                    crate::hardening::log_degradation("redis", "org_budget_check", &e.to_string());
                }
            }
        }
    }

    // Check rate limits (RPM, max_parallel_requests)
    if let Some(ref redis) = state.redis {
        match crate::auth::rate_limit::check_request_limits(
            redis,
            key_object,
            hashed_token.as_hex_str(),
            0,
        )
        .await
        {
            crate::auth::rate_limit::RateLimitResult::Allowed => {}
            crate::auth::rate_limit::RateLimitResult::RpmExceeded { limit, .. } => {
                return Err(CoreError::Auth(format!(
                    "rate limit exceeded: RPM limit is {limit}"
                )));
            }
            crate::auth::rate_limit::RateLimitResult::TpmExceeded { limit, .. } => {
                return Err(CoreError::Auth(format!(
                    "rate limit exceeded: TPM limit is {limit}"
                )));
            }
            crate::auth::rate_limit::RateLimitResult::ParallelExceeded { limit, .. } => {
                return Err(CoreError::Auth(format!(
                    "rate limit exceeded: max parallel requests is {limit}"
                )));
            }
        }
    }

    let deployments = state.router.get_all_deployments(&model);
    if deployments.is_empty() {
        return Err(CoreError::Routing(format!(
            "no deployment available for model '{model}'"
        )));
    }

    // Try each deployment in order (fallback routing)
    let mut last_error: Option<CoreError> = None;
    for (deployment_idx, deployment) in deployments.iter().enumerate() {
        let provider_model = deployment.litellm_params.model.as_str();
        let upstream_model = provider_model
            .split_once('/')
            .map_or(provider_model, |(_, model)| model);
        let custom_llm_provider = provider_model.split_once('/').map(|(provider, _)| provider);

        // Check circuit breaker for this provider
        if let Some(provider) = custom_llm_provider {
            let breaker = state.circuit_breakers.get_or_create(provider);
            if !breaker.allow_request().await {
                tracing::warn!(
                    provider = %provider,
                    event = "circuit_breaker_open",
                    "skipping deployment due to open circuit breaker"
                );
                continue;
            }
        }

        // Run pre-call guardrails
        let guardrail_context = GuardrailContext::new(CallType::Other("speech".to_string()));
        let guardrail_request = GuardrailRequest::new(body.clone());
        match state
            .guardrail_runner
            .run_pre_call(&guardrail_context, guardrail_request)
            .await
        {
            Ok((_modified_request, _report)) => {
                // Guardrails may have modified the request
            }
            Err(err) => {
                return Err(CoreError::Auth(format!(
                    "guardrail blocked request: {}",
                    err.message
                )));
            }
        }

        let request = SpeechRequest {
            model: upstream_model.to_string(),
            input: input.clone(),
            voice: voice.clone(),
            response_format: body
                .get("response_format")
                .and_then(Value::as_str)
                .map(String::from),
            speed: body.get("speed").and_then(Value::as_f64).map(|s| s as f32),
        };

        let retry_config = crate::auth::retry::RetryConfig::default();
        let mut deployment_error: Option<CoreError> = None;

        // Use network strategy for provider calls
        let max_retries = retry_config.network_strategy.max_retries;
        for attempt in 0..=max_retries {
            match speech(
                &request,
                "openai",
                deployment.litellm_params.api_base.as_deref(),
                deployment.litellm_params.api_key.clone(),
            )
            .await
            {
                Ok(response) => {
                    if let Some(provider) = custom_llm_provider {
                        let breaker = state.circuit_breakers.get_or_create(provider);
                        breaker.record_success().await;
                    }

                    if let Some(ref redis) = state.redis {
                        crate::auth::rate_limit::release_parallel_slot(
                            redis,
                            hashed_token.as_hex_str(),
                        )
                        .await;
                    }

                    // Record spend
                    record_speech_spend(state, &input, provider_model, key_object, hashed_token)
                        .await;

                    let duration = start.elapsed().as_secs_f64();
                    state
                        .metrics
                        .requests_total
                        .with_label_values(&[provider_model, "success"])
                        .inc();
                    state
                        .metrics
                        .request_duration_seconds
                        .with_label_values(&[provider_model])
                        .observe(duration);

                    tracing::info!(
                        model = %provider_model,
                        deployment_idx = deployment_idx,
                        hashed_token = %hashed_token.as_hex_str(),
                        input_length = input.len(),
                        duration_secs = duration,
                        event = "speech.success",
                        "audit: request completed successfully"
                    );

                    return Ok(AudioResponseEnum::Binary {
                        data: response.audio_data,
                        content_type: response.content_type,
                    });
                }
                Err(err) => {
                    if attempt < max_retries && crate::auth::retry::is_retryable_error(&err) {
                        deployment_error = Some(err);
                        let delay = crate::auth::retry::calculate_delay(
                            attempt + 1,
                            &retry_config.network_strategy,
                        );
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    deployment_error = Some(err);
                    break;
                }
            }
        }

        // This deployment failed after all retries
        if deployment_error
            .as_ref()
            .is_some_and(|e| e.is_upstream_failure())
            && let Some(provider) = custom_llm_provider
        {
            let breaker = state.circuit_breakers.get_or_create(provider);
            breaker.record_failure().await;
        }

        let err = deployment_error.unwrap();
        tracing::warn!(
            model = %provider_model,
            deployment_idx = deployment_idx,
            hashed_token = %hashed_token.as_hex_str(),
            error = %err,
            event = "speech.deployment_failed",
            "audit: deployment failed, trying next fallback"
        );

        last_error = Some(err);
        continue;
    }

    // All deployments exhausted
    let err = last_error.unwrap();
    let duration = start.elapsed().as_secs_f64();

    tracing::error!(
        model = %model,
        hashed_token = %hashed_token.as_hex_str(),
        error = %err,
        duration_secs = duration,
        event = "speech.all_deployments_failed",
        "audit: all deployments failed"
    );

    Err(err)
}

pub async fn run_transcription(
    state: &AppState,
    body: Value,
    _extra_headers: Option<Map<String, Value>>,
    key_object: &Arc<KeyObject>,
    hashed_token: &HashedToken,
) -> CoreResult<AudioResponseEnum> {
    let start = Instant::now();

    if !key_object.has_route_access(crate::constants::AUDIO_ROUTE_PATH_TRANSCRIPTIONS) {
        return Err(CoreError::Auth(
            "API key does not have access to this route".to_string(),
        ));
    }

    let model = body
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .ok_or_else(|| {
            CoreError::InvalidRequest("transcription body requires a model".to_string())
        })?
        .to_string();

    let file_b64 = body.get("file").and_then(Value::as_str).ok_or_else(|| {
        CoreError::InvalidRequest("transcription body requires a file (base64)".to_string())
    })?;

    let file_bytes = base64::engine::general_purpose::STANDARD
        .decode(file_b64)
        .map_err(|err| CoreError::InvalidRequest(format!("failed to decode file base64: {err}")))?;

    // Check global rate limit
    if !state.global_rate_limiter.check().await {
        return Err(CoreError::Auth("global rate limit exceeded".to_string()));
    }

    // Check model access
    if !key_object.has_model_access(&model) {
        return Err(CoreError::Auth(format!(
            "API key does not have access to model '{model}'"
        )));
    }

    // Check budget
    if !key_object.is_within_budget() {
        return Err(CoreError::Auth(format!(
            "API key has exceeded its budget limit of ${:.2}",
            key_object.max_budget.unwrap_or(0.0)
        )));
    }

    // Check team/org budget via Redis spend counters
    if let Some(ref redis) = state.redis {
        if let Some(ref team_id) = key_object.team_id {
            let mut key_buf = [0u8; 256];
            let key = spend_key(&mut key_buf, "team", team_id);
            match redis.incr_by_float(key, 0.0).await {
                Ok(team_spend) => {
                    let team_budget = state.config.team_budget;
                    if team_spend >= team_budget {
                        return Err(CoreError::Auth(format!(
                            "team '{team_id}' has exceeded its budget (${team_spend:.2} / ${team_budget:.2})"
                        )));
                    }
                }
                Err(e) => {
                    crate::hardening::log_degradation("redis", "team_budget_check", &e.to_string());
                }
            }
        }

        if let Some(ref org_id) = key_object.org_id {
            let mut key_buf = [0u8; 256];
            let key = spend_key(&mut key_buf, "org", org_id);
            match redis.incr_by_float(key, 0.0).await {
                Ok(org_spend) => {
                    let org_budget = state.config.org_budget;
                    if org_spend >= org_budget {
                        return Err(CoreError::Auth(format!(
                            "organization '{org_id}' has exceeded its budget (${org_spend:.2} / ${org_budget:.2})"
                        )));
                    }
                }
                Err(e) => {
                    crate::hardening::log_degradation("redis", "org_budget_check", &e.to_string());
                }
            }
        }
    }

    // Check rate limits (RPM, max_parallel_requests)
    if let Some(ref redis) = state.redis {
        match crate::auth::rate_limit::check_request_limits(
            redis,
            key_object,
            hashed_token.as_hex_str(),
            0,
        )
        .await
        {
            crate::auth::rate_limit::RateLimitResult::Allowed => {}
            crate::auth::rate_limit::RateLimitResult::RpmExceeded { limit, .. } => {
                return Err(CoreError::Auth(format!(
                    "rate limit exceeded: RPM limit is {limit}"
                )));
            }
            crate::auth::rate_limit::RateLimitResult::TpmExceeded { limit, .. } => {
                return Err(CoreError::Auth(format!(
                    "rate limit exceeded: TPM limit is {limit}"
                )));
            }
            crate::auth::rate_limit::RateLimitResult::ParallelExceeded { limit, .. } => {
                return Err(CoreError::Auth(format!(
                    "rate limit exceeded: max parallel requests is {limit}"
                )));
            }
        }
    }

    let deployments = state.router.get_all_deployments(&model);
    if deployments.is_empty() {
        return Err(CoreError::Routing(format!(
            "no deployment available for model '{model}'"
        )));
    }

    // Try each deployment in order (fallback routing)
    let mut last_error: Option<CoreError> = None;
    for (deployment_idx, deployment) in deployments.iter().enumerate() {
        let provider_model = deployment.litellm_params.model.as_str();
        let upstream_model = provider_model
            .split_once('/')
            .map_or(provider_model, |(_, model)| model);
        let custom_llm_provider = provider_model.split_once('/').map(|(provider, _)| provider);

        // Check circuit breaker for this provider
        if let Some(provider) = custom_llm_provider {
            let breaker = state.circuit_breakers.get_or_create(provider);
            if !breaker.allow_request().await {
                tracing::warn!(
                    provider = %provider,
                    event = "circuit_breaker_open",
                    "skipping deployment due to open circuit breaker"
                );
                continue;
            }
        }

        // Run pre-call guardrails
        let guardrail_context = GuardrailContext::new(CallType::Other("transcription".to_string()));
        let guardrail_request = GuardrailRequest::new(body.clone());
        match state
            .guardrail_runner
            .run_pre_call(&guardrail_context, guardrail_request)
            .await
        {
            Ok((_modified_request, _report)) => {
                // Guardrails may have modified the request
            }
            Err(err) => {
                return Err(CoreError::Auth(format!(
                    "guardrail blocked request: {}",
                    err.message
                )));
            }
        }

        let request = TranscriptionRequest {
            model: upstream_model.to_string(),
            file: file_bytes.clone(),
            language: body
                .get("language")
                .and_then(Value::as_str)
                .map(String::from),
            prompt: body.get("prompt").and_then(Value::as_str).map(String::from),
            response_format: body
                .get("response_format")
                .and_then(Value::as_str)
                .map(String::from),
            temperature: body
                .get("temperature")
                .and_then(Value::as_f64)
                .map(|t| t as f32),
        };

        let retry_config = crate::auth::retry::RetryConfig::default();
        let mut deployment_error: Option<CoreError> = None;

        // Use network strategy for provider calls
        let max_retries = retry_config.network_strategy.max_retries;
        for attempt in 0..=max_retries {
            match transcription(
                &request,
                "openai",
                deployment.litellm_params.api_base.as_deref(),
                deployment.litellm_params.api_key.clone(),
            )
            .await
            {
                Ok(response) => {
                    if let Some(provider) = custom_llm_provider {
                        let breaker = state.circuit_breakers.get_or_create(provider);
                        breaker.record_success().await;
                    }

                    if let Some(ref redis) = state.redis {
                        crate::auth::rate_limit::release_parallel_slot(
                            redis,
                            hashed_token.as_hex_str(),
                        )
                        .await;
                    }

                    // Record spend
                    record_transcription_spend(
                        state,
                        file_bytes.len(),
                        provider_model,
                        key_object,
                        hashed_token,
                    )
                    .await;

                    let duration = start.elapsed().as_secs_f64();
                    state
                        .metrics
                        .requests_total
                        .with_label_values(&[provider_model, "success"])
                        .inc();
                    state
                        .metrics
                        .request_duration_seconds
                        .with_label_values(&[provider_model])
                        .observe(duration);

                    tracing::info!(
                        model = %provider_model,
                        deployment_idx = deployment_idx,
                        hashed_token = %hashed_token.as_hex_str(),
                        file_size = file_bytes.len(),
                        duration_secs = duration,
                        event = "transcription.success",
                        "audit: request completed successfully"
                    );

                    return Ok(AudioResponseEnum::Json(serde_json::json!({
                        "text": response.text
                    })));
                }
                Err(err) => {
                    if attempt < max_retries && crate::auth::retry::is_retryable_error(&err) {
                        deployment_error = Some(err);
                        let delay = crate::auth::retry::calculate_delay(
                            attempt + 1,
                            &retry_config.network_strategy,
                        );
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                    deployment_error = Some(err);
                    break;
                }
            }
        }

        // This deployment failed after all retries
        if deployment_error
            .as_ref()
            .is_some_and(|e| e.is_upstream_failure())
            && let Some(provider) = custom_llm_provider
        {
            let breaker = state.circuit_breakers.get_or_create(provider);
            breaker.record_failure().await;
        }

        let err = deployment_error.unwrap();
        tracing::warn!(
            model = %provider_model,
            deployment_idx = deployment_idx,
            hashed_token = %hashed_token.as_hex_str(),
            error = %err,
            event = "transcription.deployment_failed",
            "audit: deployment failed, trying next fallback"
        );

        last_error = Some(err);
        continue;
    }

    // All deployments exhausted
    let err = last_error.unwrap();
    let duration = start.elapsed().as_secs_f64();

    tracing::error!(
        model = %model,
        hashed_token = %hashed_token.as_hex_str(),
        error = %err,
        duration_secs = duration,
        event = "transcription.all_deployments_failed",
        "audit: all deployments failed"
    );

    Err(err)
}

/// Build a spend-tracking Redis key on the stack (no heap allocation).
fn spend_key<'a>(buf: &'a mut [u8; 256], entity_type: &str, entity_id: &str) -> &'a str {
    let prefix = b"spend:";
    let mid = b":";
    let mut pos = 0;
    buf[pos..pos + prefix.len()].copy_from_slice(prefix);
    pos += prefix.len();
    let type_bytes = entity_type.as_bytes();
    buf[pos..pos + type_bytes.len()].copy_from_slice(type_bytes);
    pos += type_bytes.len();
    buf[pos..pos + mid.len()].copy_from_slice(mid);
    pos += mid.len();
    let id_bytes = entity_id.as_bytes();
    buf[pos..pos + id_bytes.len()].copy_from_slice(id_bytes);
    pos += id_bytes.len();
    unsafe { std::str::from_utf8_unchecked(&buf[..pos]) }
}

/// Calculate cost for speech and record spend entries.
async fn record_speech_spend(
    state: &AppState,
    input: &str,
    model: &str,
    key_object: &Arc<KeyObject>,
    hashed_token: &HashedToken,
) {
    let hex = hashed_token.as_hex_str();

    // Estimate cost based on input length (rough approximation)
    let char_count = input.len() as u64;
    let cost_request = CostRequest {
        model,
        usage: cost_calculator::types::Usage {
            prompt_tokens: char_count / 4, // Rough estimate: 4 chars per token
            completion_tokens: 0,
            total_tokens: char_count / 4,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        },
        custom_llm_provider: None,
        service_tier: None,
    };

    let cost = match cost_calculator::calculate_cost(&cost_request) {
        Ok(response) => response.total_cost_usd(),
        Err(ref e) => {
            tracing::warn!(error = %e, "cost calculation failed, spend not tracked");
            0.0
        }
    };

    // Record spend via worker (batched, async)
    if let Some(ref worker) = state.spend_worker {
        worker.record_update(SpendUpdateItem {
            entity_type: EntityType::Key,
            entity_id: hex.to_string(),
            cost,
        });

        if let Some(ref user_id) = key_object.user_id {
            worker.record_update(SpendUpdateItem {
                entity_type: EntityType::User,
                entity_id: user_id.clone(),
                cost,
            });
        }

        if let Some(ref team_id) = key_object.team_id {
            worker.record_update(SpendUpdateItem {
                entity_type: EntityType::Team,
                entity_id: team_id.clone(),
                cost,
            });
        }

        if let Some(ref org_id) = key_object.org_id {
            worker.record_update(SpendUpdateItem {
                entity_type: EntityType::Organization,
                entity_id: org_id.clone(),
                cost,
            });
        }
    }

    // Increment Redis spend counters directly (for real-time budget checks)
    if let Some(ref redis) = state.redis {
        let mut key_buf = [0u8; 256];
        let key = spend_key(&mut key_buf, "key", hex);
        let _ = redis.incr_by_float(key, cost).await;

        if let Some(ref user_id) = key_object.user_id {
            let key = spend_key(&mut key_buf, "user", user_id);
            let _ = redis.incr_by_float(key, cost).await;
        }
        if let Some(ref team_id) = key_object.team_id {
            let key = spend_key(&mut key_buf, "team", team_id);
            let _ = redis.incr_by_float(key, cost).await;
        }
        if let Some(ref org_id) = key_object.org_id {
            let key = spend_key(&mut key_buf, "org", org_id);
            let _ = redis.incr_by_float(key, cost).await;
        }
    }
}

/// Calculate cost for transcription and record spend entries.
async fn record_transcription_spend(
    state: &AppState,
    file_size: usize,
    model: &str,
    key_object: &Arc<KeyObject>,
    hashed_token: &HashedToken,
) {
    let hex = hashed_token.as_hex_str();

    // Estimate cost based on file size (rough approximation: 1 minute per 1MB)
    let minutes = (file_size as f64 / 1_000_000.0).ceil() as u64;
    let cost_request = CostRequest {
        model,
        usage: cost_calculator::types::Usage {
            prompt_tokens: minutes * 60, // Rough estimate: 60 tokens per second
            completion_tokens: 0,
            total_tokens: minutes * 60,
            prompt_tokens_details: None,
            completion_tokens_details: None,
        },
        custom_llm_provider: None,
        service_tier: None,
    };

    let cost = match cost_calculator::calculate_cost(&cost_request) {
        Ok(response) => response.total_cost_usd(),
        Err(ref e) => {
            tracing::warn!(error = %e, "cost calculation failed, spend not tracked");
            0.0
        }
    };

    // Record spend via worker (batched, async)
    if let Some(ref worker) = state.spend_worker {
        worker.record_update(SpendUpdateItem {
            entity_type: EntityType::Key,
            entity_id: hex.to_string(),
            cost,
        });

        if let Some(ref user_id) = key_object.user_id {
            worker.record_update(SpendUpdateItem {
                entity_type: EntityType::User,
                entity_id: user_id.clone(),
                cost,
            });
        }

        if let Some(ref team_id) = key_object.team_id {
            worker.record_update(SpendUpdateItem {
                entity_type: EntityType::Team,
                entity_id: team_id.clone(),
                cost,
            });
        }

        if let Some(ref org_id) = key_object.org_id {
            worker.record_update(SpendUpdateItem {
                entity_type: EntityType::Organization,
                entity_id: org_id.clone(),
                cost,
            });
        }
    }

    // Increment Redis spend counters directly (for real-time budget checks)
    if let Some(ref redis) = state.redis {
        let mut key_buf = [0u8; 256];
        let key = spend_key(&mut key_buf, "key", hex);
        let _ = redis.incr_by_float(key, cost).await;

        if let Some(ref user_id) = key_object.user_id {
            let key = spend_key(&mut key_buf, "user", user_id);
            let _ = redis.incr_by_float(key, cost).await;
        }
        if let Some(ref team_id) = key_object.team_id {
            let key = spend_key(&mut key_buf, "team", team_id);
            let _ = redis.incr_by_float(key, cost).await;
        }
        if let Some(ref org_id) = key_object.org_id {
            let key = spend_key(&mut key_buf, "org", org_id);
            let _ = redis.incr_by_float(key, cost).await;
        }
    }
}
