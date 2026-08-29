use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use futures::Stream;
use litellm_core::auth::{HashedToken, KeyObject};
use litellm_core::chat_completions::{chat_completions, chat_completions_stream, StreamingChunk};
use litellm_core::chat_completions::types::{ChatCompletionsRequest, ChatCompletionsResponse};
use litellm_core::cost_calculator::{self, CostRequest};
use litellm_core::persistence::CacheStore;
use litellm_core::spend_tracking::{EntityType, SpendUpdateItem};
use litellm_core::{CoreError, CoreResult};
use serde_json::{Map, Value};

use crate::state::AppState;

// Global request counter for zero-alloc request IDs
static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a zero-alloc request ID using atomic counter
fn next_request_id() -> u64 {
    REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Build a spend-tracking Redis key on the stack (no heap allocation).
/// Format: `spend:<entity_type>:<entity_id>`
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

/// Response from chat completions - either a full response or a streaming stream
pub enum ChatCompletionsResult {
    /// Non-streaming: full response (serialized JSON bytes)
    Complete(Vec<u8>),
    /// Streaming: stream of SSE chunks
    Streaming(Pin<Box<dyn Stream<Item = StreamingChunk> + Send>>),
}

pub async fn run(
    state: &AppState,
    body: Value,
    extra_headers: Option<Map<String, Value>>,
    key_object: &Arc<KeyObject>,
    hashed_token: &HashedToken,
) -> CoreResult<ChatCompletionsResult> {
    let start = Instant::now();
    let request_id = next_request_id();

    let model = body
        .get("model")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .ok_or_else(|| {
            CoreError::InvalidRequest("chat completions body requires a model".to_string())
        })?;

    tracing::info!(
        request_id = request_id,
        model = %model,
        hashed_token = %hashed_token.as_hex_str(),
        event = "chat_completions.start",
        "audit: request started"
    );

    // Check global rate limit
    if !state.global_rate_limiter.check().await {
        return Err(CoreError::Auth("global rate limit exceeded".to_string()));
    }

    // Check model access
    if !key_object.has_model_access(model) {
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
                    // Fail open: allow request if Redis is down
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
                    // Fail open: allow request if Redis is down
                }
            }
        }
    }

    // Input validation
    validate_chat_completions_body(&body)?;

    // Check rate limits (RPM, max_parallel_requests)
    if let Some(ref redis) = state.redis {
        match crate::auth::rate_limit::check_request_limits(redis, key_object, hashed_token.as_hex_str()).await {
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

    let deployment = state
        .router
        .get_available_deployment(model)
        .ok_or_else(|| {
            CoreError::Routing(format!("no deployment available for model '{model}'"))
        })?;
    let provider_model = deployment.litellm_params.model.as_str();
    let upstream_model = provider_model
        .split_once('/')
        .map_or(provider_model, |(_, model)| model);
    let custom_llm_provider = provider_model.split_once('/').map(|(provider, _)| provider);
    
    // Check circuit breaker for this provider
    if let Some(provider) = custom_llm_provider {
        let breaker = state.circuit_breakers.get_or_create(provider);
        if !breaker.allow_request().await {
            return Err(CoreError::Network(format!(
                "circuit breaker is open for provider '{provider}'"
            )));
        }
    }
    
    let timeout = body
        .get("timeout")
        .and_then(Value::as_f64)
        .map(Duration::from_secs_f64)
        .or_else(|| Some(Duration::from_secs_f64(state.config.default_request_timeout_secs)));

    // Check if streaming is requested
    let is_streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    
    let mut body = body;
    body.as_object_mut()
        .ok_or_else(|| {
            CoreError::InvalidRequest("chat completions body must be an object".to_string())
        })?
        .insert(
            "model".to_string(),
            Value::String(upstream_model.to_string()),
        );

    // Build cache key for non-streaming requests (hash raw bytes, avoid Value tree)
    let cache_key = if !is_streaming {
        use sha2::Digest;
        let mut hasher = sha2::Sha256::new();
        hasher.update(provider_model.as_bytes());
        hasher.update(b":");
        if let Some(messages) = body.get("messages") {
            if let Ok(bytes) = serde_json::to_vec(messages) {
                hasher.update(bytes);
            }
        }
        let hash = hasher.finalize();
        let mut hex_str = String::with_capacity(64 + 11);
        hex_str.push_str("cache:chat:");
        for b in hash.iter() {
            use std::fmt::Write;
            write!(hex_str, "{b:02x}").unwrap();
        }
        Some(hex_str)
    } else {
        None
    };

    // Check cache before upstream call
    if let (Some(key), Some(redis)) = (&cache_key, &state.redis) {
        use litellm_core::persistence::CacheStore;
        if let Ok(Some(cached)) = redis.get(key).await {
            state.metrics.requests_total.with_label_values(&[provider_model, "cached"]).inc();
            // Serialize cached Value to bytes
            let cached_bytes = serde_json::to_vec(&cached).map_err(|err| {
                CoreError::InvalidResponse(format!("failed to serialize cached response: {err}"))
            })?;
            return Ok(ChatCompletionsResult::Complete(cached_bytes));
        }
    }

    // Handle streaming vs non-streaming
    if is_streaming {
        let request = ChatCompletionsRequest {
            model: provider_model,
            messages: body
                .get("messages")
                .cloned()
                .unwrap_or(Value::Array(vec![])),
            optional_params: body
                .as_object()
                .map(|obj| {
                    obj.iter()
                        .filter(|(k, _)| k.as_str() != "model" && k.as_str() != "messages")
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect()
                })
                .unwrap_or_default(),
            api_key: deployment.litellm_params.api_key.as_deref(),
            api_base: deployment.litellm_params.api_base.as_deref(),
            custom_llm_provider,
            extra_headers,
            timeout,
        };

        let stream = chat_completions_stream(request);
        
        match stream {
            Ok(stream) => {
                if let Some(provider) = custom_llm_provider {
                    let breaker = state.circuit_breakers.get_or_create(provider);
                    breaker.record_success().await;
                }
                state.metrics.requests_total.with_label_values(&[provider_model, "success"]).inc();
                state.metrics.request_duration_seconds.with_label_values(&[provider_model]).observe(start.elapsed().as_secs_f64());
                Ok(ChatCompletionsResult::Streaming(stream))
            }
            Err(err) => {
                if let Some(provider) = custom_llm_provider {
                    let breaker = state.circuit_breakers.get_or_create(provider);
                    breaker.record_failure().await;
                }
                state.metrics.requests_total.with_label_values(&[provider_model, "error"]).inc();
                state.metrics.request_duration_seconds.with_label_values(&[provider_model]).observe(start.elapsed().as_secs_f64());
                Err(err)
            }
        }
    } else {
        let retry_config = crate::auth::retry::RetryConfig::default();

        // Build request factory - called once per attempt (only on retry)
        let build_request = || ChatCompletionsRequest {
            model: provider_model,
            messages: body
                .get("messages")
                .cloned()
                .unwrap_or(Value::Array(vec![])),
            optional_params: body
                .as_object()
                .map(|obj| {
                    obj.iter()
                        .filter(|(k, _)| k.as_str() != "model" && k.as_str() != "messages")
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect()
                })
                .unwrap_or_default(),
            api_key: deployment.litellm_params.api_key.as_deref(),
            api_base: deployment.litellm_params.api_base.as_deref(),
            custom_llm_provider,
            extra_headers: extra_headers.clone(),
            timeout,
        };

        // First attempt - no clone needed
        let mut request = build_request();
        let mut last_error: Option<CoreError> = None;

        for attempt in 0..=retry_config.max_retries {
            match chat_completions(request).await {
                Ok(response) => {
                    if let Some(provider) = custom_llm_provider {
                        let breaker = state.circuit_breakers.get_or_create(provider);
                        breaker.record_success().await;
                    }

                    if let Some(ref redis) = state.redis {
                        crate::auth::rate_limit::release_parallel_slot(redis, hashed_token.as_hex_str()).await;
                    }

                    if let Some(ref redis) = state.redis {
                        crate::auth::rate_limit::check_token_limits(
                            redis,
                            key_object,
                            hashed_token.as_hex_str(),
                            response.usage.prompt_tokens,
                            response.usage.completion_tokens,
                        )
                        .await;
                    }

                    record_spend(state, &response, provider_model, key_object, hashed_token).await;

                    let duration = start.elapsed().as_secs_f64();
                    state.metrics.requests_total.with_label_values(&[provider_model, "success"]).inc();
                    state.metrics.request_duration_seconds.with_label_values(&[provider_model]).observe(duration);
                    state.metrics.tokens_total.with_label_values(&[provider_model, "prompt"]).inc_by(response.usage.prompt_tokens as u64);
                    state.metrics.tokens_total.with_label_values(&[provider_model, "completion"]).inc_by(response.usage.completion_tokens as u64);

                    tracing::info!(
                        request_id = request_id,
                        model = %provider_model,
                        hashed_token = %hashed_token.as_hex_str(),
                        prompt_tokens = response.usage.prompt_tokens,
                        completion_tokens = response.usage.completion_tokens,
                        duration_secs = duration,
                        event = "chat_completions.success",
                        "audit: request completed successfully"
                    );

                    // Serialize directly to bytes (avoid Value intermediate)
                    let response_bytes = serde_json::to_vec(&response).map_err(|err| {
                        CoreError::InvalidResponse(format!(
                            "failed to serialize chat completions response: {err}"
                        ))
                    })?;

                    // Store in cache (convert bytes to Value for cache storage)
                    if let (Some(key), Some(redis)) = (&cache_key, &state.redis) {
                        use litellm_core::persistence::CacheStore;
                        let cache_ttl = state.config.cache_ttl_secs;
                        // Parse bytes back to Value for cache (cache API requires Value)
                        if let Ok(response_value) = serde_json::from_slice::<Value>(&response_bytes) {
                            let _ = redis.set(key, &response_value, Some(cache_ttl)).await;
                        }
                    }

                    return Ok(ChatCompletionsResult::Complete(response_bytes));
                }
                Err(err) => {
                    // Check if we should retry
                    if attempt < retry_config.max_retries && crate::auth::retry::is_retryable_error(&err) {
                        last_error = Some(err);
                        let delay = crate::auth::retry::calculate_delay(attempt + 1, &retry_config);
                        tokio::time::sleep(delay).await;
                        // Only clone on actual retry
                        request = build_request();
                        continue;
                    }
                    // Non-retryable or out of attempts
                    last_error = Some(err);
                    break;
                }
            }
        }

        // All attempts exhausted or non-retryable error
        let err = last_error.unwrap();

        if let Some(provider) = custom_llm_provider {
            let breaker = state.circuit_breakers.get_or_create(provider);
            breaker.record_failure().await;
        }

        let duration = start.elapsed().as_secs_f64();
        state.metrics.requests_total.with_label_values(&[provider_model, "error"]).inc();
        state.metrics.request_duration_seconds.with_label_values(&[provider_model]).observe(duration);

        tracing::warn!(
            request_id = request_id,
            model = %provider_model,
            hashed_token = %hashed_token.as_hex_str(),
            error = %err,
            duration_secs = duration,
            event = "chat_completions.error",
            "audit: request failed"
        );

        Err(err)
    }
}

/// Calculate cost from the response and record spend entries.
async fn record_spend(
    state: &AppState,
    response: &ChatCompletionsResponse,
    model: &str,
    key_object: &Arc<KeyObject>,
    hashed_token: &HashedToken,
) {
    let hex = hashed_token.as_hex_str();
    let usage = &response.usage;

    // Build cost request (zero-alloc: borrows model &str)
    let cost_request = CostRequest {
        model,
        usage: cost_calculator::types::Usage {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            prompt_tokens_details: Some(cost_calculator::types::PromptTokensDetails {
                cached_tokens: usage.prompt_tokens_details.cached_tokens,
                cache_hit_tokens: usage.prompt_tokens_details.cached_tokens,
                cache_creation_tokens: usage.prompt_tokens_details.cache_creation_tokens,
                text_tokens: usage.prompt_tokens_details.text_tokens,
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

fn validate_chat_completions_body(body: &Value) -> CoreResult<()> {
    let messages = body.get("messages").ok_or_else(|| {
        CoreError::InvalidRequest("chat completions body requires 'messages'".to_string())
    })?;

    let messages_arr = messages.as_array().ok_or_else(|| {
        CoreError::InvalidRequest("'messages' must be an array".to_string())
    })?;

    if messages_arr.is_empty() {
        return Err(CoreError::InvalidRequest(
            "'messages' must not be empty".to_string(),
        ));
    }

    for (i, msg) in messages_arr.iter().enumerate() {
        if msg.get("role").and_then(Value::as_str).is_none() {
            return Err(CoreError::InvalidRequest(format!(
                "messages[{i}] requires a 'role' field"
            )));
        }
    }

    if let Some(temp) = body.get("temperature").and_then(Value::as_f64) {
        if !(0.0..=2.0).contains(&temp) {
            return Err(CoreError::InvalidRequest(format!(
                "'temperature' must be between 0 and 2, got {temp}"
            )));
        }
    }

    if let Some(top_p) = body.get("top_p").and_then(Value::as_f64) {
        if !(0.0..=1.0).contains(&top_p) {
            return Err(CoreError::InvalidRequest(format!(
                "'top_p' must be between 0 and 1, got {top_p}"
            )));
        }
    }

    if let Some(max_tokens) = body.get("max_tokens").and_then(Value::as_i64) {
        if max_tokens <= 0 {
            return Err(CoreError::InvalidRequest(format!(
                "'max_tokens' must be positive, got {max_tokens}"
            )));
        }
    }

    if let Some(stream) = body.get("stream") {
        if !stream.is_boolean() {
            return Err(CoreError::InvalidRequest(
                "'stream' must be a boolean".to_string(),
            ));
        }
    }

    Ok(())
}
