use std::sync::Arc;
use std::time::{Duration, Instant};

use litellm_core::auth::{HashedToken, KeyObject};
use litellm_core::constants::ANTHROPIC_MESSAGES_PROVIDER;
use litellm_core::cost_calculator::{self, CostRequest};
use litellm_core::messages::types::{AnthropicMessagesResponse, MessagesRequest};
use litellm_core::messages::{messages, messages_stream};
use litellm_core::persistence::CacheStore;
use litellm_core::spend_tracking::{EntityType, SpendUpdateItem};
use litellm_core::{CoreError, CoreResult};
use serde_json::{Map, Value};

use crate::integrations::custom_guardrail::{GuardrailContext, GuardrailRequest};
use crate::integrations::custom_logger::CallType;
use crate::state::AppState;

pub(crate) enum MessagesResponse {
    Json(Value),
    Stream(reqwest::Response),
}

pub async fn run(
    state: &AppState,
    body: Value,
    extra_headers: Option<Map<String, Value>>,
    key_object: &Arc<KeyObject>,
    hashed_token: &HashedToken,
) -> CoreResult<MessagesResponse> {
    let start = Instant::now();

    let model = body
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .ok_or_else(|| CoreError::InvalidRequest("messages body requires a model".to_string()))?
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

    let timeout = Some(Duration::from_secs_f64(
        state.config.default_request_timeout_secs,
    ));
    let mut body = body;

    // Try each deployment in order (fallback routing)
    let mut last_error: Option<CoreError> = None;
    for (deployment_idx, deployment) in deployments.iter().enumerate() {
        let provider_model = deployment.litellm_params.model.as_str();
        let upstream_model = provider_model
            .split_once('/')
            .map_or(provider_model, |(_, model)| model);
        let custom_llm_provider = if provider_model.contains('/') {
            None
        } else {
            Some(ANTHROPIC_MESSAGES_PROVIDER)
        };

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

        body.as_object_mut()
            .ok_or_else(|| {
                CoreError::InvalidRequest("messages body must be an object".to_string())
            })?
            .insert(
                "model".to_string(),
                Value::String(upstream_model.to_string()),
            );

        // Run pre-call guardrails
        let guardrail_context = GuardrailContext::new(CallType::Completion);
        let guardrail_request = GuardrailRequest::new(body.clone());
        match state
            .guardrail_runner
            .run_pre_call(&guardrail_context, guardrail_request)
            .await
        {
            Ok((modified_request, _report)) => {
                body = modified_request.data;
            }
            Err(err) => {
                return Err(CoreError::Auth(format!(
                    "guardrail blocked request: {}",
                    err.message
                )));
            }
        }

        let request = MessagesRequest {
            model: provider_model,
            body: body.clone(),
            api_key: deployment.litellm_params.api_key.as_deref(),
            api_base: deployment.litellm_params.api_base.as_deref(),
            custom_llm_provider,
            extra_headers: extra_headers.clone(),
            timeout,
        };

        let is_streaming = request.body.get("stream").and_then(Value::as_bool) == Some(true);

        if is_streaming {
            match messages_stream(request).await {
                Ok(stream_response) => {
                    if let Some(provider) = custom_llm_provider {
                        let breaker = state.circuit_breakers.get_or_create(provider);
                        breaker.record_success().await;
                    }

                    // Note: Anthropic streaming spend tracking would need to parse message_start event
                    // For now, we'll handle non-streaming spend tracking fully
                    // Streaming spend tracking for messages can be added similarly to chat_completions

                    return Ok(MessagesResponse::Stream(stream_response));
                }
                Err(err) => {
                    if let Some(provider) = custom_llm_provider {
                        let breaker = state.circuit_breakers.get_or_create(provider);
                        breaker.record_failure().await;
                    }
                    last_error = Some(err);
                    continue;
                }
            }
        } else {
            let retry_config = crate::auth::retry::RetryConfig::default();
            let mut request = request;
            let mut deployment_error: Option<CoreError> = None;

            // Use network strategy for provider calls
            let max_retries = retry_config.network_strategy.max_retries;
            for attempt in 0..=max_retries {
                match messages(request).await {
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

                        // Parse usage for spend tracking and metrics
                        let (input_tokens, output_tokens) =
                            if let Some(usage_value) = &response.usage {
                                let input = usage_value
                                    .get("input_tokens")
                                    .and_then(Value::as_u64)
                                    .unwrap_or(0);
                                let output = usage_value
                                    .get("output_tokens")
                                    .and_then(Value::as_u64)
                                    .unwrap_or(0);
                                (input, output)
                            } else {
                                (0, 0)
                            };

                        // Record spend
                        record_spend(state, &response, provider_model, key_object, hashed_token)
                            .await;

                        if let Some(ref redis) = state.redis {
                            crate::auth::rate_limit::check_token_limits(
                                redis,
                                key_object,
                                hashed_token.as_hex_str(),
                                input_tokens,
                                output_tokens,
                            )
                            .await;
                        }

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
                        state
                            .metrics
                            .tokens_total
                            .with_label_values(&[provider_model, "prompt"])
                            .inc_by(input_tokens);
                        state
                            .metrics
                            .tokens_total
                            .with_label_values(&[provider_model, "completion"])
                            .inc_by(output_tokens);

                        tracing::info!(
                            model = %provider_model,
                            deployment_idx = deployment_idx,
                            hashed_token = %hashed_token.as_hex_str(),
                            input_tokens = input_tokens,
                            output_tokens = output_tokens,
                            duration_secs = duration,
                            event = "messages.success",
                            "audit: request completed successfully"
                        );

                        return serde_json::to_value(response)
                            .map(MessagesResponse::Json)
                            .map_err(|err| {
                                CoreError::InvalidResponse(format!(
                                    "failed to serialize messages response: {err}"
                                ))
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
                            request = MessagesRequest {
                                model: provider_model,
                                body: body.clone(),
                                api_key: deployment.litellm_params.api_key.as_deref(),
                                api_base: deployment.litellm_params.api_base.as_deref(),
                                custom_llm_provider,
                                extra_headers: extra_headers.clone(),
                                timeout,
                            };
                            continue;
                        }
                        deployment_error = Some(err);
                        break;
                    }
                }
            }

            // This deployment failed after all retries
            if let Some(provider) = custom_llm_provider {
                let breaker = state.circuit_breakers.get_or_create(provider);
                breaker.record_failure().await;
            }

            let err = deployment_error.unwrap();
            tracing::warn!(
                model = %provider_model,
                deployment_idx = deployment_idx,
                hashed_token = %hashed_token.as_hex_str(),
                error = %err,
                event = "messages.deployment_failed",
                "audit: deployment failed, trying next fallback"
            );

            last_error = Some(err);
            continue;
        }
    }

    // All deployments exhausted
    let err = last_error.unwrap();
    let duration = start.elapsed().as_secs_f64();

    tracing::error!(
        model = %model,
        hashed_token = %hashed_token.as_hex_str(),
        error = %err,
        duration_secs = duration,
        event = "messages.all_deployments_failed",
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

/// Calculate cost from the response and record spend entries.
async fn record_spend(
    state: &AppState,
    response: &AnthropicMessagesResponse,
    model: &str,
    key_object: &Arc<KeyObject>,
    hashed_token: &HashedToken,
) {
    let hex = hashed_token.as_hex_str();

    // Parse usage from the Value type
    let (input_tokens, output_tokens) = if let Some(usage_value) = &response.usage {
        let input = usage_value
            .get("input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let output = usage_value
            .get("output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        (input, output)
    } else {
        (0, 0)
    };

    // Build cost request
    let cost_request = CostRequest {
        model,
        usage: cost_calculator::types::Usage {
            prompt_tokens: input_tokens,
            completion_tokens: output_tokens,
            total_tokens: input_tokens + output_tokens,
            prompt_tokens_details: Some(cost_calculator::types::PromptTokensDetails {
                cached_tokens: 0,
                cache_hit_tokens: 0,
                cache_creation_tokens: 0,
                text_tokens: input_tokens,
                audio_tokens: 0,
                image_tokens: 0,
            }),
            completion_tokens_details: None,
        },
        custom_llm_provider: None,
        service_tier: None,
    };

    let cost = match cost_calculator::calculate_cost(&cost_request) {
        Ok(response) => response.total_cost_usd(),
        Err(_) => 0.0,
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
