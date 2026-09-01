use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use futures::Stream;
use litellm_core::auth::{HashedToken, KeyObject};
use litellm_core::chat_completions::types::{ChatCompletionsRequest, ChatCompletionsResponse};
use litellm_core::chat_completions::{StreamingChunk, chat_completions, chat_completions_stream};
use litellm_core::cost_calculator::{self, CostRequest};
use litellm_core::persistence::CacheStore;
use litellm_core::spend_tracking::{EntityType, SpendUpdateItem};
use litellm_core::{CoreError, CoreResult};
use serde_json::{Map, Value};

use crate::integrations::custom_guardrail::{GuardrailContext, GuardrailRequest};
use crate::integrations::custom_logger::CallType;
use crate::integrations::custom_logger::{
    CallbackTiming, CallbackValue, CustomLoggerRunner, ModelCallDetails,
};
use crate::state::AppState;

// Global request counter for zero-alloc request IDs
static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a zero-alloc request ID using atomic counter
fn next_request_id() -> u64 {
    REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// A stream wrapper that tracks usage and records spend after completion.
/// Clones necessary data so it can record spend independently after the stream ends.
struct SpendTrackingStream {
    inner: Pin<Box<dyn Stream<Item = StreamingChunk> + Send>>,
    tracker: crate::streaming::StreamingCostTracker,
    tool_call_tracker: crate::streaming::ToolCallAccumulator,
    finish_reason_tracker: crate::streaming::FinishReasonTracker,
    thinking_tracker: crate::streaming::ThinkingBlockTracker,
    role_stripper: crate::streaming::RoleStripper,
    timeout_enforcer: crate::streaming::StreamTimeoutEnforcer,
    repetition_detector: crate::streaming::ModelRepetitionDetector,
    provider_feature_handler: crate::streaming::ProviderFeatureHandler,
    comprehensive_timeout_tracker: crate::streaming::ComprehensiveTimeoutTracker,
    /// Provider-specific fields extracted from streaming chunks (e.g., Anthropic thinking, OpenAI metadata)
    provider_specific_fields: Option<Map<String, Value>>,
    state: Arc<AppState>,
    model: String,
    key_object: Arc<KeyObject>,
    hashed_token_hex: String,
}

impl Stream for SpendTrackingStream {
    type Item = StreamingChunk;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Check if stream has exceeded maximum duration
        if let Some(error_msg) = self.timeout_enforcer.check_timeout() {
            return Poll::Ready(Some(Err(CoreError::Timeout(error_msg))));
        }

        // Check comprehensive timeouts (connect, read, pool, total, chunk)
        if let Some(error_msg) = self.comprehensive_timeout_tracker.check_timeouts() {
            return Poll::Ready(Some(Err(CoreError::Timeout(error_msg))));
        }

        match self.inner.as_mut().poll_next(cx) {
            Poll::Ready(Some(Ok(Some(chunk)))) => {
                // Update comprehensive timeout tracker
                self.comprehensive_timeout_tracker.update_chunk_time();
                if !self.comprehensive_timeout_tracker.connection_established {
                    self.comprehensive_timeout_tracker.connection_established = true;
                }

                // Check for model repetition
                if let Some(error_msg) = self.repetition_detector.check_repetition(&chunk) {
                    return Poll::Ready(Some(Err(CoreError::InvalidResponse(error_msg))));
                }

                // Strip role from delta after first chunk
                let chunk = self.role_stripper.process_chunk(&chunk);

                // Track usage for cost calculation
                self.tracker.accumulate(&chunk);

                // Track tool calls and function calls
                if let Some(choices) = chunk.get("choices").and_then(|c| c.as_array())
                    && let Some(first_choice) = choices.first()
                    && let Some(delta) = first_choice.get("delta")
                {
                    self.tool_call_tracker.accumulate_tool_call_delta(delta);
                    self.tool_call_tracker.accumulate_function_call_delta(delta);
                    self.thinking_tracker.process_chunk(&chunk);
                }

                // Track finish reason
                self.finish_reason_tracker.process_chunk(&chunk);

                // Extract and preserve provider-specific fields
                if let Some(fields) = crate::streaming::extract_provider_specific_fields(&chunk) {
                    // Merge with existing fields (later chunks override earlier ones)
                    if let Some(ref mut existing) = self.provider_specific_fields {
                        existing.extend(fields);
                    } else {
                        self.provider_specific_fields = Some(fields);
                    }
                }

                // Extract provider-specific features from chunk
                self.provider_feature_handler
                    .extract_from_chunk(&chunk, None);

                Poll::Ready(Some(Ok(Some(chunk))))
            }
            Poll::Ready(Some(Ok(None))) => {
                // End of stream - spawn background task to record spend
                let tracker = &self.tracker;
                if tracker.prompt_tokens > 0 || tracker.completion_tokens > 0 {
                    let state = Arc::clone(&self.state);
                    let model = self.model.clone();
                    let key_object = Arc::clone(&self.key_object);
                    let hashed_token_hex = self.hashed_token_hex.clone();
                    let prompt_tokens = tracker.prompt_tokens;
                    let completion_tokens = tracker.completion_tokens;
                    let total_tokens = tracker.total_tokens;
                    let cached_tokens = tracker.cached_tokens;
                    let cache_creation_tokens = tracker.cache_creation_tokens;

                    tokio::spawn(async move {
                        // Calculate cost
                        let cost_request = CostRequest {
                            model: &model,
                            usage: cost_calculator::types::Usage {
                                prompt_tokens,
                                completion_tokens,
                                total_tokens,
                                prompt_tokens_details: Some(
                                    cost_calculator::types::PromptTokensDetails {
                                        cached_tokens,
                                        cache_hit_tokens: cached_tokens,
                                        cache_creation_tokens,
                                        text_tokens: prompt_tokens
                                            .saturating_sub(cached_tokens + cache_creation_tokens),
                                        audio_tokens: 0,
                                        image_tokens: 0,
                                    },
                                ),
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
                            worker
                                .record_update(SpendUpdateItem {
                                    entity_type: EntityType::Key,
                                    entity_id: hashed_token_hex.clone(),
                                    cost,
                                })
                                .await;

                            if let Some(ref user_id) = key_object.user_id {
                                worker
                                    .record_update(SpendUpdateItem {
                                        entity_type: EntityType::User,
                                        entity_id: user_id.clone(),
                                        cost,
                                    })
                                    .await;
                            }

                            if let Some(ref team_id) = key_object.team_id {
                                worker
                                    .record_update(SpendUpdateItem {
                                        entity_type: EntityType::Team,
                                        entity_id: team_id.clone(),
                                        cost,
                                    })
                                    .await;
                            }

                            if let Some(ref org_id) = key_object.org_id {
                                worker
                                    .record_update(SpendUpdateItem {
                                        entity_type: EntityType::Organization,
                                        entity_id: org_id.clone(),
                                        cost,
                                    })
                                    .await;
                            }
                        }

                        // Increment Redis spend counters
                        if let Some(ref redis) = state.redis {
                            let mut key_buf = [0u8; 256];
                            let key = spend_key(&mut key_buf, "key", &hashed_token_hex);
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

                        // Update metrics
                        state
                            .metrics
                            .tokens_total
                            .with_label_values(&[&model, "prompt"])
                            .inc_by(prompt_tokens);
                        state
                            .metrics
                            .tokens_total
                            .with_label_values(&[&model, "completion"])
                            .inc_by(completion_tokens);

                        tracing::info!(
                            model = %model,
                            hashed_token = %hashed_token_hex,
                            prompt_tokens = prompt_tokens,
                            completion_tokens = completion_tokens,
                            event = "chat_completions.streaming.complete",
                            "audit: streaming request completed"
                        );
                    });
                }
                Poll::Ready(Some(Ok(None)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
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

    if !key_object.has_route_access(crate::constants::CHAT_COMPLETIONS_ROUTE_PATH) {
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
            CoreError::InvalidRequest("chat completions body requires a model".to_string())
        })?
        .to_string();

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

    // Check rate limits (RPM, TPM, max_parallel_requests)
    if let Some(ref redis) = state.redis {
        let estimated_tokens = body
            .get("messages")
            .and_then(|m| m.as_array())
            .map(|msgs| {
                let chars: usize = msgs
                    .iter()
                    .filter_map(|m| m.get("content").and_then(|c| c.as_str()))
                    .map(|s| s.len())
                    .sum();
                (chars / 4) as u64
            })
            .unwrap_or(0);
        match crate::auth::rate_limit::check_request_limits(
            redis,
            key_object,
            hashed_token.as_hex_str(),
            estimated_tokens,
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

    tracing::debug!(
        request_id = request_id,
        model = %model,
        "Looking up deployments for model"
    );

    let deployments = state.router.get_all_deployments(&model);

    tracing::debug!(
        request_id = request_id,
        model = %model,
        deployments_found = deployments.len(),
        "Found {} deployments",
        deployments.len()
    );

    if deployments.is_empty() {
        return Err(CoreError::Routing(format!(
            "no deployment available for model '{model}'"
        )));
    }

    let timeout = body
        .get("timeout")
        .and_then(Value::as_f64)
        .map(Duration::from_secs_f64)
        .or_else(|| {
            Some(Duration::from_secs_f64(
                state.config.default_request_timeout_secs,
            ))
        });

    // Check if streaming is requested
    let is_streaming = body.get("stream").and_then(Value::as_bool).unwrap_or(false);

    let mut body = body;

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

        body.as_object_mut()
            .ok_or_else(|| {
                CoreError::InvalidRequest("chat completions body must be an object".to_string())
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
                // Guardrails may have modified the request
                body = modified_request.data;
            }
            Err(err) => {
                return Err(CoreError::Auth(format!(
                    "guardrail blocked request: {}",
                    err.message
                )));
            }
        }

        // Build cache key for non-streaming requests (hash raw bytes, avoid Value tree)
        let cache_key = if !is_streaming {
            use sha2::Digest;
            let mut hasher = sha2::Sha256::new();
            hasher.update(provider_model.as_bytes());
            hasher.update(b":");
            if let Some(messages) = body.get("messages")
                && let Ok(bytes) = serde_json::to_vec(messages)
            {
                hasher.update(bytes);
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
                state
                    .metrics
                    .requests_total
                    .with_label_values(&[provider_model, "cached"])
                    .inc();
                let cached_bytes = serde_json::to_vec(&cached).map_err(|err| {
                    CoreError::InvalidResponse(format!(
                        "failed to serialize cached response: {err}"
                    ))
                })?;
                return Ok(ChatCompletionsResult::Complete(cached_bytes));
            }
        }

        // Handle streaming vs non-streaming
        let result = if is_streaming {
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
                extra_headers: extra_headers.clone(),
                timeout,
            };

            let stream = chat_completions_stream(request);

            match stream {
                Ok(stream) => {
                    if let Some(provider) = custom_llm_provider {
                        let breaker = state.circuit_breakers.get_or_create(provider);
                        breaker.record_success().await;
                    }
                    state
                        .metrics
                        .requests_total
                        .with_label_values(&[provider_model, "success"])
                        .inc();
                    state
                        .metrics
                        .request_duration_seconds
                        .with_label_values(&[provider_model])
                        .observe(start.elapsed().as_secs_f64());

                    // Count images in request for vision/multimodal support
                    let messages = body
                        .get("messages")
                        .cloned()
                        .unwrap_or(Value::Array(vec![]));
                    let (image_count, image_tokens) =
                        crate::streaming::StreamingCostTracker::count_images_in_messages(&messages);

                    let mut tracker = crate::streaming::StreamingCostTracker::new();
                    tracker.image_count = image_count;
                    tracker.image_tokens = image_tokens;

                    // Extract provider-specific features from request
                    let optional_params = body
                        .as_object()
                        .map(|obj| {
                            obj.iter()
                                .filter(|(k, _)| k.as_str() != "model" && k.as_str() != "messages")
                                .map(|(k, v)| (k.clone(), v.clone()))
                                .collect()
                        })
                        .unwrap_or_default();
                    let mut provider_feature_handler =
                        crate::streaming::ProviderFeatureHandler::new();
                    provider_feature_handler
                        .extract_from_request(&Value::Object(optional_params), custom_llm_provider);

                    let tracking_stream = SpendTrackingStream {
                        inner: stream,
                        tracker,
                        tool_call_tracker: crate::streaming::ToolCallAccumulator::new(),
                        finish_reason_tracker: crate::streaming::FinishReasonTracker::new(),
                        thinking_tracker: crate::streaming::ThinkingBlockTracker::new(),
                        role_stripper: crate::streaming::RoleStripper::new(),
                        timeout_enforcer: crate::streaming::StreamTimeoutEnforcer::new(
                            std::env::var("LITELLM_MAX_STREAMING_DURATION_SECONDS")
                                .ok()
                                .and_then(|v| v.parse().ok()),
                        ),
                        repetition_detector: crate::streaming::ModelRepetitionDetector::new(
                            std::env::var("LITELLM_MAX_STREAMING_REPETITIONS")
                                .ok()
                                .and_then(|v| v.parse().ok())
                                .unwrap_or(10),
                            5,
                        ),
                        provider_feature_handler,
                        comprehensive_timeout_tracker:
                            crate::streaming::ComprehensiveTimeoutTracker::new(
                                crate::streaming::ComprehensiveTimeoutConfig::from_env(),
                            ),
                        provider_specific_fields: None,
                        state: Arc::new(state.clone()),
                        model: provider_model.to_string(),
                        key_object: Arc::clone(key_object),
                        hashed_token_hex: hashed_token.as_hex_str().to_string(),
                    };
                    Ok(ChatCompletionsResult::Streaming(Box::pin(tracking_stream)))
                }
                Err(err) => {
                    if let Some(provider) = custom_llm_provider
                        && err.is_upstream_failure()
                    {
                        let breaker = state.circuit_breakers.get_or_create(provider);
                        breaker.record_failure().await;
                    }
                    state
                        .metrics
                        .requests_total
                        .with_label_values(&[provider_model, "error"])
                        .inc();
                    state
                        .metrics
                        .request_duration_seconds
                        .with_label_values(&[provider_model])
                        .observe(start.elapsed().as_secs_f64());
                    Err(err)
                }
            }
        } else {
            let retry_config = crate::auth::retry::RetryConfig::default();

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

            let mut request = build_request();
            let mut deployment_error: Option<CoreError> = None;

            // Use network strategy for provider calls
            let max_retries = retry_config.network_strategy.max_retries;
            for attempt in 0..=max_retries {
                match chat_completions(request).await {
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

                        if let Some(ref redis) = state.redis {
                            crate::auth::rate_limit::check_token_limits(
                                redis,
                                key_object,
                                hashed_token.as_hex_str(),
                                0,
                                response.usage.completion_tokens,
                            )
                            .await;
                        }

                        record_spend(state, &response, provider_model, key_object, hashed_token)
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
                        state
                            .metrics
                            .tokens_total
                            .with_label_values(&[provider_model, "prompt"])
                            .inc_by(response.usage.prompt_tokens);
                        state
                            .metrics
                            .tokens_total
                            .with_label_values(&[provider_model, "completion"])
                            .inc_by(response.usage.completion_tokens);

                        tracing::info!(
                            request_id = request_id,
                            model = %provider_model,
                            deployment_idx = deployment_idx,
                            hashed_token = %hashed_token.as_hex_str(),
                            prompt_tokens = response.usage.prompt_tokens,
                            completion_tokens = response.usage.completion_tokens,
                            duration_secs = duration,
                            event = "chat_completions.success",
                            "audit: request completed successfully"
                        );

                        let response_bytes = serde_json::to_vec(&response).map_err(|err| {
                            CoreError::InvalidResponse(format!(
                                "failed to serialize chat completions response: {err}"
                            ))
                        })?;

                        if let (Some(key), Some(redis)) = (&cache_key, &state.redis) {
                            use litellm_core::persistence::CacheStore;
                            let cache_ttl = state.config.cache_ttl_secs;
                            if let Ok(response_value) =
                                serde_json::from_slice::<Value>(&response_bytes)
                            {
                                let _ = redis.set(key, &response_value, Some(cache_ttl)).await;
                            }
                        }

                        // Execute callbacks
                        let callback_details = ModelCallDetails::new(
                            provider_model,
                            custom_llm_provider.unwrap_or("unknown"),
                            CallType::ChatCompletion,
                        );
                        let callback_response = CallbackValue::new(
                            "chat.completion",
                            serde_json::to_value(&response).unwrap_or(Value::Null),
                        );
                        let callback_timing = CallbackTiming::new(
                            start.elapsed().as_secs_f64() - duration,
                            start.elapsed().as_secs_f64(),
                        );
                        let runner = CustomLoggerRunner::new(state.loggers.as_ref().clone());
                        let _ = runner
                            .async_log_success_event(
                                &callback_details,
                                &callback_response,
                                callback_timing,
                            )
                            .await;

                        return Ok(ChatCompletionsResult::Complete(response_bytes));
                    }
                    Err(err) => {
                        if attempt < max_retries && crate::auth::retry::is_retryable_error(&err) {
                            deployment_error = Some(err);
                            let delay = crate::auth::retry::calculate_delay(
                                attempt + 1,
                                &retry_config.network_strategy,
                            );
                            tokio::time::sleep(delay).await;
                            request = build_request();
                            continue;
                        }
                        deployment_error = Some(err);
                        break;
                    }
                }
            }

            // This deployment failed after all retries
            if let Some(provider) = custom_llm_provider
                && let Some(ref err) = deployment_error
                && err.is_upstream_failure()
            {
                let breaker = state.circuit_breakers.get_or_create(provider);
                breaker.record_failure().await;
            }

            let duration = start.elapsed().as_secs_f64();
            state
                .metrics
                .requests_total
                .with_label_values(&[provider_model, "error"])
                .inc();
            state
                .metrics
                .request_duration_seconds
                .with_label_values(&[provider_model])
                .observe(duration);

            let err = match deployment_error {
                Some(e) => e,
                None => {
                    // This happens when all deployments were skipped (e.g., circuit breaker open)
                    tracing::warn!(
                        request_id = request_id,
                        model = %provider_model,
                        deployment_idx = deployment_idx,
                        hashed_token = %hashed_token.as_hex_str(),
                        duration_secs = duration,
                        event = "chat_completions.deployment_skipped",
                        "audit: deployment skipped (likely circuit breaker open)"
                    );
                    continue; // Try next deployment
                }
            };

            tracing::warn!(
                request_id = request_id,
                model = %provider_model,
                deployment_idx = deployment_idx,
                hashed_token = %hashed_token.as_hex_str(),
                error = %err,
                duration_secs = duration,
                event = "chat_completions.deployment_failed",
                "audit: deployment failed, trying next fallback"
            );

            // Execute failure callbacks
            let callback_details = ModelCallDetails::new(
                provider_model,
                custom_llm_provider.unwrap_or("unknown"),
                CallType::ChatCompletion,
            );
            let callback_timing = CallbackTiming::new(
                start.elapsed().as_secs_f64() - duration,
                start.elapsed().as_secs_f64(),
            );
            let runner = CustomLoggerRunner::new(state.loggers.as_ref().clone());
            let _ = runner
                .async_log_failure_event(&callback_details, None, callback_timing)
                .await;

            last_error = Some(err);
            continue; // Try next deployment
        };

        // For streaming, return immediately (success or error)
        return result;
    }

    // All deployments exhausted
    let err = last_error.unwrap_or_else(|| {
        CoreError::Routing(
            "no chat completions deployment is configured for this model".to_string(),
        )
    });
    let duration = start.elapsed().as_secs_f64();

    tracing::error!(
        request_id = request_id,
        model = %model,
        hashed_token = %hashed_token.as_hex_str(),
        error = %err,
        duration_secs = duration,
        event = "chat_completions.all_deployments_failed",
        "audit: all deployments failed"
    );

    // Execute final failure callbacks
    let callback_details = ModelCallDetails::new(&model, "unknown", CallType::ChatCompletion);
    let callback_timing = CallbackTiming::new(
        start.elapsed().as_secs_f64() - duration,
        start.elapsed().as_secs_f64(),
    );
    let runner = CustomLoggerRunner::new(state.loggers.as_ref().clone());
    let _ = runner
        .async_log_failure_event(&callback_details, None, callback_timing)
        .await;

    Err(err)
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
        Err(ref e) => {
            tracing::warn!(error = %e, "cost calculation failed, spend not tracked");
            0.0
        }
    };

    // Record spend via worker (batched, async)
    if let Some(ref worker) = state.spend_worker {
        worker
            .record_update(SpendUpdateItem {
                entity_type: EntityType::Key,
                entity_id: hex.to_string(),
                cost,
            })
            .await;

        if let Some(ref user_id) = key_object.user_id {
            worker
                .record_update(SpendUpdateItem {
                    entity_type: EntityType::User,
                    entity_id: user_id.clone(),
                    cost,
                })
                .await;
        }

        if let Some(ref team_id) = key_object.team_id {
            worker
                .record_update(SpendUpdateItem {
                    entity_type: EntityType::Team,
                    entity_id: team_id.clone(),
                    cost,
                })
                .await;
        }

        if let Some(ref org_id) = key_object.org_id {
            worker
                .record_update(SpendUpdateItem {
                    entity_type: EntityType::Organization,
                    entity_id: org_id.clone(),
                    cost,
                })
                .await;
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

    let messages_arr = messages
        .as_array()
        .ok_or_else(|| CoreError::InvalidRequest("'messages' must be an array".to_string()))?;

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

    if let Some(temp) = body.get("temperature").and_then(Value::as_f64)
        && !(0.0..=2.0).contains(&temp)
    {
        return Err(CoreError::InvalidRequest(format!(
            "'temperature' must be between 0 and 2, got {temp}"
        )));
    }

    if let Some(top_p) = body.get("top_p").and_then(Value::as_f64)
        && !(0.0..=1.0).contains(&top_p)
    {
        return Err(CoreError::InvalidRequest(format!(
            "'top_p' must be between 0 and 1, got {top_p}"
        )));
    }

    if let Some(max_tokens) = body.get("max_tokens").and_then(Value::as_i64)
        && max_tokens <= 0
    {
        return Err(CoreError::InvalidRequest(format!(
            "'max_tokens' must be positive, got {max_tokens}"
        )));
    }

    if let Some(stream) = body.get("stream")
        && !stream.is_boolean()
    {
        return Err(CoreError::InvalidRequest(
            "'stream' must be a boolean".to_string(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_streaming_cost_tracker_accumulate_usage() {
        let mut tracker = crate::streaming::StreamingCostTracker::new();

        // First chunk with no usage
        let chunk1 = json!({
            "id": "chatcmpl-123",
            "choices": [{
                "delta": {"content": "Hello"},
                "index": 0
            }]
        });
        tracker.accumulate(&chunk1);
        assert_eq!(tracker.prompt_tokens, 0);
        assert_eq!(tracker.completion_tokens, 0);

        // Final chunk with usage
        let chunk2 = json!({
            "id": "chatcmpl-123",
            "choices": [{
                "delta": {},
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "prompt_tokens_details": {
                    "cached_tokens": 5,
                    "cache_creation_tokens": 2
                }
            }
        });
        tracker.accumulate(&chunk2);

        assert_eq!(tracker.prompt_tokens, 10);
        assert_eq!(tracker.completion_tokens, 20);
        assert_eq!(tracker.total_tokens, 30);
        assert_eq!(tracker.cached_tokens, 5);
        assert_eq!(tracker.cache_creation_tokens, 2);
    }

    #[test]
    fn test_streaming_cost_tracker_multiple_usage_chunks() {
        let mut tracker = crate::streaming::StreamingCostTracker::new();

        // Some providers send usage in multiple chunks
        let chunk1 = json!({
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 0,
                "total_tokens": 15
            }
        });
        tracker.accumulate(&chunk1);
        assert_eq!(tracker.prompt_tokens, 15);
        assert_eq!(tracker.completion_tokens, 0);

        let chunk2 = json!({
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 25,
                "total_tokens": 40
            }
        });
        tracker.accumulate(&chunk2);
        assert_eq!(tracker.prompt_tokens, 15);
        assert_eq!(tracker.completion_tokens, 25);
        assert_eq!(tracker.total_tokens, 40);
    }

    #[test]
    fn test_streaming_cost_tracker_no_prompt_details() {
        let mut tracker = crate::streaming::StreamingCostTracker::new();

        let chunk = json!({
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        });
        tracker.accumulate(&chunk);

        assert_eq!(tracker.prompt_tokens, 10);
        assert_eq!(tracker.completion_tokens, 20);
        assert_eq!(tracker.cached_tokens, 0);
        assert_eq!(tracker.cache_creation_tokens, 0);
    }

    #[tokio::test]
    async fn test_guardrail_blocks_request() {
        use crate::integrations::custom_guardrail::{
            CustomGuardrail, CustomGuardrailRunner, GuardrailDecision, GuardrailError,
            GuardrailEventHook, GuardrailFuture,
        };
        use std::sync::Arc;

        struct BlockingGuardrail;

        impl CustomGuardrail for BlockingGuardrail {
            fn guardrail_name(&self) -> &str {
                "blocking_guardrail"
            }

            fn supported_event_hooks(&self) -> &[GuardrailEventHook] {
                &[GuardrailEventHook::PreCall]
            }

            fn async_pre_call_hook<'a>(
                &'a self,
                _context: &'a GuardrailContext,
                request: GuardrailRequest,
            ) -> GuardrailFuture<'a> {
                Box::pin(async move {
                    if request
                        .data
                        .get("messages")
                        .and_then(|m| m.as_array())
                        .map(|arr| {
                            arr.iter().any(|msg| {
                                msg.get("content")
                                    .and_then(|c| c.as_str())
                                    .map(|s| s.contains("blocked_keyword"))
                                    .unwrap_or(false)
                            })
                        })
                        .unwrap_or(false)
                    {
                        Err(GuardrailError::blocked("request contains blocked keyword"))
                    } else {
                        Ok(GuardrailDecision::Allow(request))
                    }
                })
            }
        }

        let guardrail_runner = Arc::new(CustomGuardrailRunner::new(vec![Arc::new(
            BlockingGuardrail,
        )]));
        let context = GuardrailContext::new(CallType::Completion);

        // Test blocking
        let blocked_request = GuardrailRequest::new(json!({
            "messages": [{"role": "user", "content": "this contains blocked_keyword"}]
        }));
        let result = guardrail_runner
            .run_pre_call(&context, blocked_request)
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("blocked keyword"));

        // Test allowing
        let allowed_request = GuardrailRequest::new(json!({
            "messages": [{"role": "user", "content": "this is fine"}]
        }));
        let result = guardrail_runner
            .run_pre_call(&context, allowed_request)
            .await;
        assert!(result.is_ok());
    }
}
