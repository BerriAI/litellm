//! Streaming chunk processing enhancements.
//!
//! Provides enhanced streaming chunk processing including:
//! - Function/tool call parsing and accumulation
//! - Thinking block handling
//! - Finish reason tracking
//! - Provider-specific field preservation
//! - Role stripping from delta after first chunk
//! - Comprehensive timeout handling

use serde_json::{Map, Value};
use std::time::Duration;

/// Strips role from delta after first chunk.
/// First chunk includes role: "assistant", subsequent chunks have role stripped.
/// Handles Mistral's None role.
#[derive(Debug, Default)]
pub struct RoleStripper {
    sent_first_chunk: bool,
}

impl RoleStripper {
    pub fn new() -> Self {
        Self {
            sent_first_chunk: false,
        }
    }

    /// Process a chunk and strip role from delta if not first chunk.
    /// Returns a modified chunk with role stripped if appropriate.
    pub fn process_chunk(&mut self, chunk: &Value) -> Value {
        let mut modified_chunk = chunk.clone();

        if let Some(choices) = modified_chunk
            .get_mut("choices")
            .and_then(|c| c.as_array_mut())
            && let Some(first_choice) = choices.first_mut()
            && let Some(delta) = first_choice.get_mut("delta")
        {
            if self.sent_first_chunk {
                // Strip role from subsequent chunks
                if let Some(delta_obj) = delta.as_object_mut() {
                    delta_obj.remove("role");
                }
            } else {
                // First chunk - keep role but mark as sent
                // Handle Mistral's None role by setting it to "assistant"
                if let Some(delta_obj) = delta.as_object_mut()
                    && let Some(role) = delta_obj.get("role")
                    && role.is_null()
                {
                    delta_obj.insert("role".to_string(), Value::String("assistant".to_string()));
                }
                self.sent_first_chunk = true;
            }
        }

        modified_chunk
    }

    /// Check if we've sent the first chunk
    pub fn has_sent_first_chunk(&self) -> bool {
        self.sent_first_chunk
    }

    /// Reset the stripper for a new stream
    pub fn reset(&mut self) {
        self.sent_first_chunk = false;
    }
}

/// Enforces maximum streaming duration to prevent hanging streams.
/// Tracks stream start time and raises error if duration exceeded.
#[derive(Debug)]
pub struct StreamTimeoutEnforcer {
    start_time: std::time::Instant,
    max_duration_secs: Option<u64>,
}

impl StreamTimeoutEnforcer {
    pub fn new(max_duration_secs: Option<u64>) -> Self {
        Self {
            start_time: std::time::Instant::now(),
            max_duration_secs,
        }
    }

    /// Check if the stream has exceeded the maximum duration.
    /// Returns error message if exceeded, None otherwise.
    pub fn check_timeout(&self) -> Option<String> {
        if let Some(max_secs) = self.max_duration_secs {
            let elapsed = self.start_time.elapsed().as_secs();
            if elapsed > max_secs {
                return Some(format!(
                    "Stream exceeded maximum duration of {} seconds (elapsed: {} seconds)",
                    max_secs, elapsed
                ));
            }
        }
        None
    }

    /// Get the elapsed time in seconds
    pub fn elapsed_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Reset the enforcer for a new stream
    pub fn reset(&mut self) {
        self.start_time = std::time::Instant::now();
    }
}

impl Default for StreamTimeoutEnforcer {
    fn default() -> Self {
        Self::new(None)
    }
}

/// Comprehensive timeout configuration for streaming requests.
/// Supports separate timeouts for different phases of the request lifecycle.
#[derive(Debug, Clone)]
pub struct ComprehensiveTimeoutConfig {
    /// Timeout for establishing the connection (DNS, TCP, TLS)
    pub connect_timeout: Option<Duration>,
    /// Timeout for reading each chunk (idle timeout between chunks)
    pub read_timeout: Option<Duration>,
    /// Timeout for waiting for a connection from the pool
    pub pool_timeout: Option<Duration>,
    /// Total timeout for the entire request lifecycle
    pub total_timeout: Option<Duration>,
    /// Maximum time between chunks (idle timeout)
    pub chunk_timeout: Option<Duration>,
}

impl ComprehensiveTimeoutConfig {
    pub fn new() -> Self {
        Self {
            connect_timeout: None,
            read_timeout: None,
            pool_timeout: None,
            total_timeout: None,
            chunk_timeout: None,
        }
    }

    /// Load timeout configuration from environment variables.
    pub fn from_env() -> Self {
        Self {
            connect_timeout: std::env::var("LITELLM_CONNECT_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .map(Duration::from_secs),
            read_timeout: std::env::var("LITELLM_READ_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .map(Duration::from_secs),
            pool_timeout: std::env::var("LITELLM_POOL_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .map(Duration::from_secs),
            total_timeout: std::env::var("LITELLM_TOTAL_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .map(Duration::from_secs),
            chunk_timeout: std::env::var("LITELLM_CHUNK_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .map(Duration::from_secs),
        }
    }

    /// Check if any timeout is configured.
    pub fn has_any_timeout(&self) -> bool {
        self.connect_timeout.is_some()
            || self.read_timeout.is_some()
            || self.pool_timeout.is_some()
            || self.total_timeout.is_some()
            || self.chunk_timeout.is_some()
    }
}

impl Default for ComprehensiveTimeoutConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

/// Comprehensive timeout tracker for streaming requests.
/// Tracks different timeout phases and enforces timeout limits.
#[derive(Debug)]
pub struct ComprehensiveTimeoutTracker {
    config: ComprehensiveTimeoutConfig,
    start_time: std::time::Instant,
    last_chunk_time: std::time::Instant,
    pub connection_established: bool,
}

impl ComprehensiveTimeoutTracker {
    pub fn new(config: ComprehensiveTimeoutConfig) -> Self {
        let now = std::time::Instant::now();
        Self {
            config,
            start_time: now,
            last_chunk_time: now,
            connection_established: false,
        }
    }

    /// Mark that the connection has been established.
    pub fn mark_connection_established(&mut self) {
        self.connection_established = true;
    }

    /// Update the last chunk time (call this when receiving a chunk).
    pub fn update_chunk_time(&mut self) {
        self.last_chunk_time = std::time::Instant::now();
    }

    /// Check all timeout conditions and return error message if any timeout is exceeded.
    pub fn check_timeouts(&self) -> Option<String> {
        let elapsed = self.start_time.elapsed();
        let time_since_last_chunk = self.last_chunk_time.elapsed();

        // Check total timeout
        if let Some(total_timeout) = self.config.total_timeout
            && elapsed > total_timeout
        {
            return Some(format!(
                "Total request timeout exceeded: {}s elapsed, limit is {}s",
                elapsed.as_secs(),
                total_timeout.as_secs()
            ));
        }

        // Check chunk timeout (idle timeout between chunks)
        if let Some(chunk_timeout) = self.config.chunk_timeout
            && self.connection_established
            && time_since_last_chunk > chunk_timeout
        {
            return Some(format!(
                "Chunk timeout exceeded: {}s since last chunk, limit is {}s",
                time_since_last_chunk.as_secs(),
                chunk_timeout.as_secs()
            ));
        }

        // Check read timeout (similar to chunk timeout but for initial read)
        if let Some(read_timeout) = self.config.read_timeout
            && self.connection_established
            && time_since_last_chunk > read_timeout
        {
            return Some(format!(
                "Read timeout exceeded: {}s since last activity, limit is {}s",
                time_since_last_chunk.as_secs(),
                read_timeout.as_secs()
            ));
        }

        None
    }

    /// Get elapsed time since request start.
    pub fn elapsed_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Get time since last chunk.
    pub fn time_since_last_chunk_secs(&self) -> u64 {
        self.last_chunk_time.elapsed().as_secs()
    }
}

impl Default for ComprehensiveTimeoutTracker {
    fn default() -> Self {
        Self::new(ComprehensiveTimeoutConfig::default())
    }
}

/// Detects model repetition to prevent infinite loops.
/// Tracks last N chunks and raises error if same chunk repeated too many times.
#[derive(Debug)]
pub struct ModelRepetitionDetector {
    /// Last N chunks for comparison
    recent_chunks: Vec<String>,
    /// Maximum number of recent chunks to track
    max_history: usize,
    /// Number of times current chunk has been repeated
    repetition_count: usize,
    /// Maximum allowed repetitions before raising error
    max_repetitions: usize,
}

impl ModelRepetitionDetector {
    pub fn new(max_repetitions: usize, max_history: usize) -> Self {
        Self {
            recent_chunks: Vec::with_capacity(max_history),
            max_history,
            repetition_count: 0,
            max_repetitions,
        }
    }

    /// Check if the chunk is a repetition and update tracking.
    /// Returns error message if repetition limit exceeded, None otherwise.
    pub fn check_repetition(&mut self, chunk: &Value) -> Option<String> {
        // Extract content from chunk for comparison
        let chunk_content = self.extract_chunk_content(chunk);

        // Skip empty chunks or chunks with no content
        if chunk_content.is_empty() {
            return None;
        }

        // Check if this chunk matches the most recent chunk
        if let Some(last_chunk) = self.recent_chunks.last() {
            if *last_chunk == chunk_content {
                self.repetition_count += 1;

                if self.repetition_count >= self.max_repetitions {
                    return Some(format!(
                        "Model is repeating the same chunk {} times: {:?}",
                        self.repetition_count,
                        chunk_content.chars().take(100).collect::<String>()
                    ));
                }
            } else {
                // Reset counter if chunk is different
                self.repetition_count = 0;
            }
        }

        // Add chunk to history
        self.recent_chunks.push(chunk_content);

        // Keep only the most recent chunks
        if self.recent_chunks.len() > self.max_history {
            self.recent_chunks.remove(0);
        }

        None
    }

    /// Extract content from chunk for comparison.
    /// Focuses on the delta content to detect repetition.
    fn extract_chunk_content(&self, chunk: &Value) -> String {
        let mut content = String::new();

        // Extract delta content from choices
        if let Some(choices) = chunk.get("choices").and_then(|c| c.as_array())
            && let Some(first_choice) = choices.first()
            && let Some(delta) = first_choice.get("delta")
        {
            // Extract text content
            if let Some(text) = delta.get("content").and_then(|c| c.as_str()) {
                content.push_str(text);
            }

            // Extract tool calls
            if let Some(tool_calls) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                for tool_call in tool_calls {
                    if let Some(function) = tool_call.get("function")
                        && let Some(args) = function.get("arguments").and_then(|a| a.as_str())
                    {
                        content.push_str(args);
                    }
                }
            }
        }

        content
    }

    /// Reset the detector for a new stream
    pub fn reset(&mut self) {
        self.recent_chunks.clear();
        self.repetition_count = 0;
    }

    /// Get the current repetition count
    pub fn get_repetition_count(&self) -> usize {
        self.repetition_count
    }
}

impl Default for ModelRepetitionDetector {
    fn default() -> Self {
        // Default: 10 repetitions max, track last 5 chunks
        Self::new(10, 5)
    }
}

/// Handles provider-specific features for OpenAI, Anthropic, and Bedrock.
/// Extracts and preserves provider-specific fields from streaming chunks.
#[derive(Debug, Default)]
pub struct ProviderFeatureHandler {
    /// OpenAI: response_format (JSON mode, JSON schema)
    pub openai_response_format: Option<Value>,
    /// Anthropic: cache_control and prompt caching
    pub anthropic_cache_control: Option<Value>,
    /// Bedrock: guardrailConfig
    pub bedrock_guardrail_config: Option<Value>,
    /// Bedrock: performanceConfig
    pub bedrock_performance_config: Option<Value>,
}

impl ProviderFeatureHandler {
    pub fn new() -> Self {
        Self {
            openai_response_format: None,
            anthropic_cache_control: None,
            bedrock_guardrail_config: None,
            bedrock_performance_config: None,
        }
    }

    /// Extract provider-specific features from request optional_params.
    pub fn extract_from_request(&mut self, optional_params: &Value, provider: Option<&str>) {
        match provider {
            Some("openai") | Some("azure") => {
                // Extract OpenAI response_format
                if let Some(response_format) = optional_params.get("response_format") {
                    self.openai_response_format = Some(response_format.clone());
                }
            }
            Some("anthropic") => {
                // Extract Anthropic cache_control
                if let Some(cache_control) = optional_params.get("cache_control") {
                    self.anthropic_cache_control = Some(cache_control.clone());
                }
            }
            Some("bedrock") | Some("bedrock_converse") => {
                // Extract Bedrock guardrailConfig
                if let Some(guardrail_config) = optional_params.get("guardrailConfig") {
                    self.bedrock_guardrail_config = Some(guardrail_config.clone());
                }
                // Extract Bedrock performanceConfig
                if let Some(performance_config) = optional_params.get("performanceConfig") {
                    self.bedrock_performance_config = Some(performance_config.clone());
                }
            }
            _ => {}
        }
    }

    /// Extract provider-specific features from streaming chunk.
    pub fn extract_from_chunk(&mut self, chunk: &Value, provider: Option<&str>) {
        // Look for provider-specific fields in the chunk
        if let Some(provider_fields) = chunk.get("provider_specific_fields") {
            match provider {
                Some("openai") | Some("azure") => {
                    if let Some(response_format) = provider_fields.get("response_format") {
                        self.openai_response_format = Some(response_format.clone());
                    }
                }
                Some("anthropic") => {
                    if let Some(cache_control) = provider_fields.get("cache_control") {
                        self.anthropic_cache_control = Some(cache_control.clone());
                    }
                }
                Some("bedrock") | Some("bedrock_converse") => {
                    if let Some(guardrail_config) = provider_fields.get("guardrailConfig") {
                        self.bedrock_guardrail_config = Some(guardrail_config.clone());
                    }
                    if let Some(performance_config) = provider_fields.get("performanceConfig") {
                        self.bedrock_performance_config = Some(performance_config.clone());
                    }
                }
                _ => {}
            }
        }
    }

    /// Check if JSON mode is enabled (OpenAI)
    pub fn is_json_mode(&self) -> bool {
        if let Some(ref response_format) = self.openai_response_format
            && let Some(format_type) = response_format.get("type").and_then(Value::as_str)
        {
            return format_type == "json_object" || format_type == "json_schema";
        }
        false
    }

    /// Check if prompt caching is enabled (Anthropic)
    pub fn is_cache_enabled(&self) -> bool {
        self.anthropic_cache_control.is_some()
    }

    /// Check if guardrails are enabled (Bedrock)
    pub fn is_guardrail_enabled(&self) -> bool {
        self.bedrock_guardrail_config.is_some()
    }

    /// Get all provider-specific features as a Value for logging/tracking
    pub fn to_value(&self) -> Value {
        let mut features = serde_json::Map::new();

        if let Some(ref response_format) = self.openai_response_format {
            features.insert(
                "openai_response_format".to_string(),
                response_format.clone(),
            );
        }
        if let Some(ref cache_control) = self.anthropic_cache_control {
            features.insert("anthropic_cache_control".to_string(), cache_control.clone());
        }
        if let Some(ref guardrail_config) = self.bedrock_guardrail_config {
            features.insert(
                "bedrock_guardrail_config".to_string(),
                guardrail_config.clone(),
            );
        }
        if let Some(ref performance_config) = self.bedrock_performance_config {
            features.insert(
                "bedrock_performance_config".to_string(),
                performance_config.clone(),
            );
        }

        Value::Object(features)
    }
}

/// Tracks tool call state across streaming chunks.
/// Accumulates partial JSON arguments as they stream in.
#[derive(Debug, Default)]
pub struct ToolCallAccumulator {
    /// Tool calls indexed by their position in the tool_calls array
    tool_calls: Vec<ToolCallState>,
}

#[derive(Debug, Clone)]
struct ToolCallState {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
    call_type: String,
}

impl ToolCallAccumulator {
    pub fn new() -> Self {
        Self {
            tool_calls: Vec::new(),
        }
    }

    /// Process a tool call delta from a streaming chunk.
    /// Handles partial arguments and None values (Azure, Mistral).
    pub fn accumulate_tool_call_delta(&mut self, delta: &Value) {
        if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for tool_call in tool_calls {
                let index = tool_call.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;

                // Ensure we have space for this index
                while self.tool_calls.len() <= index {
                    self.tool_calls.push(ToolCallState {
                        id: None,
                        name: None,
                        arguments: String::new(),
                        call_type: "function".to_string(),
                    });
                }

                let state = &mut self.tool_calls[index];

                // Accumulate ID if present
                if let Some(id) = tool_call.get("id").and_then(Value::as_str) {
                    state.id = Some(id.to_string());
                }

                // Accumulate type if present
                if let Some(call_type) = tool_call.get("type").and_then(Value::as_str) {
                    state.call_type = call_type.to_string();
                }

                // Accumulate function details
                if let Some(function) = tool_call.get("function") {
                    // Accumulate name if present
                    if let Some(name) = function.get("name").and_then(Value::as_str) {
                        state.name = Some(name.to_string());
                    }

                    // Accumulate arguments (handle None for Azure/Mistral)
                    if let Some(args) = function.get("arguments")
                        && let Some(args_str) = args.as_str()
                    {
                        state.arguments.push_str(args_str);
                    }
                    // If arguments is null, we just skip it (Azure/Mistral behavior)
                }
            }
        }
    }

    /// Process a function call delta from a streaming chunk (legacy OpenAI format).
    pub fn accumulate_function_call_delta(&mut self, delta: &Value) {
        if let Some(function_call) = delta.get("function_call") {
            // Ensure we have at least one tool call state
            if self.tool_calls.is_empty() {
                self.tool_calls.push(ToolCallState {
                    id: None,
                    name: None,
                    arguments: String::new(),
                    call_type: "function".to_string(),
                });
            }

            let state = &mut self.tool_calls[0];

            // Accumulate name if present
            if let Some(name) = function_call.get("name").and_then(Value::as_str) {
                state.name = Some(name.to_string());
            }

            // Accumulate arguments (handle None for Azure/Mistral)
            if let Some(args) = function_call.get("arguments")
                && let Some(args_str) = args.as_str()
            {
                state.arguments.push_str(args_str);
            }
        }
    }

    /// Check if we have any accumulated tool calls
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
            && self
                .tool_calls
                .iter()
                .any(|tc| tc.name.is_some() || !tc.arguments.is_empty())
    }

    /// Get the accumulated tool calls as a JSON value
    pub fn to_json(&self) -> Option<Value> {
        if !self.has_tool_calls() {
            return None;
        }

        let tool_calls: Vec<Value> = self
            .tool_calls
            .iter()
            .filter(|tc| tc.name.is_some() || !tc.arguments.is_empty())
            .map(|tc| {
                let mut tool_call = Map::new();
                if let Some(ref id) = tc.id {
                    tool_call.insert("id".to_string(), Value::String(id.clone()));
                }
                tool_call.insert("type".to_string(), Value::String(tc.call_type.clone()));

                let mut function = Map::new();
                if let Some(ref name) = tc.name {
                    function.insert("name".to_string(), Value::String(name.clone()));
                }
                function.insert("arguments".to_string(), Value::String(tc.arguments.clone()));
                tool_call.insert("function".to_string(), Value::Object(function));

                Value::Object(tool_call)
            })
            .collect();

        Some(Value::Array(tool_calls))
    }
}

/// Tracks finish reason across streaming chunks.
/// Handles the complex logic of stripping finish_reason from content chunks
/// and emitting it on trailing empty-delta chunks.
#[derive(Debug, Default)]
pub struct FinishReasonTracker {
    received_finish_reason: Option<String>,
    intermittent_finish_reason: Option<String>,
    has_emitted_finish_reason: bool,
}

impl FinishReasonTracker {
    pub fn new() -> Self {
        Self {
            received_finish_reason: None,
            intermittent_finish_reason: None,
            has_emitted_finish_reason: false,
        }
    }

    /// Process a chunk and extract finish reason if present.
    /// Returns the finish reason if it should be emitted with this chunk.
    pub fn process_chunk(&mut self, chunk: &Value) -> Option<String> {
        // Extract finish_reason from choices
        if let Some(choices) = chunk.get("choices").and_then(Value::as_array)
            && let Some(first_choice) = choices.first()
        {
            // Check for finish_reason in the choice
            if let Some(finish_reason) = first_choice.get("finish_reason").and_then(Value::as_str) {
                self.intermittent_finish_reason = Some(finish_reason.to_string());

                // Check if delta is empty (trailing chunk)
                let delta = first_choice.get("delta");
                let is_empty_delta = delta
                    .map(|d| {
                        let has_content = d
                            .get("content")
                            .and_then(Value::as_str)
                            .map(|s| !s.is_empty())
                            .unwrap_or(false);
                        let has_tool_calls = d
                            .get("tool_calls")
                            .and_then(Value::as_array)
                            .map(|a| !a.is_empty())
                            .unwrap_or(false);
                        let has_function_call = d.get("function_call").is_some();
                        !has_content && !has_tool_calls && !has_function_call
                    })
                    .unwrap_or(true);

                if is_empty_delta && !self.has_emitted_finish_reason {
                    self.received_finish_reason = Some(finish_reason.to_string());
                    self.has_emitted_finish_reason = true;
                    return self.received_finish_reason.clone();
                }
            }
        }
        None
    }

    /// Get the final finish reason
    pub fn get_finish_reason(&self) -> Option<&str> {
        self.received_finish_reason
            .as_deref()
            .or(self.intermittent_finish_reason.as_deref())
    }

    /// Check if we've emitted the finish reason
    pub fn has_emitted_finish_reason(&self) -> bool {
        self.has_emitted_finish_reason
    }
}

/// Tracks thinking/reasoning blocks for Anthropic extended thinking.
#[derive(Debug, Default)]
pub struct ThinkingBlockTracker {
    sent_first_thinking_block: bool,
    sent_last_thinking_block: bool,
    thinking_content: String,
}

impl ThinkingBlockTracker {
    pub fn new() -> Self {
        Self {
            sent_first_thinking_block: false,
            sent_last_thinking_block: false,
            thinking_content: String::new(),
        }
    }

    /// Process a chunk and extract thinking/reasoning content.
    pub fn process_chunk(&mut self, chunk: &Value) {
        if let Some(choices) = chunk.get("choices").and_then(Value::as_array)
            && let Some(first_choice) = choices.first()
        {
            if let Some(delta) = first_choice.get("delta") {
                // Check for thinking_blocks
                if let Some(thinking_blocks) =
                    delta.get("thinking_blocks").and_then(Value::as_array)
                {
                    for block in thinking_blocks {
                        if let Some(thinking) = block.get("thinking").and_then(Value::as_str) {
                            self.thinking_content.push_str(thinking);
                            if !self.sent_first_thinking_block {
                                self.sent_first_thinking_block = true;
                            }
                        }
                    }
                }

                // Check for reasoning_content
                if let Some(reasoning) = delta.get("reasoning_content").and_then(Value::as_str) {
                    self.thinking_content.push_str(reasoning);
                    if !self.sent_first_thinking_block {
                        self.sent_first_thinking_block = true;
                    }
                }
            }

            // Check if this is the last chunk (has finish_reason)
            if first_choice.get("finish_reason").is_some() {
                self.sent_last_thinking_block = true;
            }
        }
    }

    /// Get the accumulated thinking content
    pub fn get_thinking_content(&self) -> &str {
        &self.thinking_content
    }

    /// Check if we've sent the first thinking block
    pub fn has_sent_first_thinking_block(&self) -> bool {
        self.sent_first_thinking_block
    }

    /// Check if we've sent the last thinking block
    pub fn has_sent_last_thinking_block(&self) -> bool {
        self.sent_last_thinking_block
    }
}

/// Tracks usage across streaming chunks with support for provider-reported costs.
/// Handles OpenAI's stream_options.include_usage format and provider-specific cost reporting.
#[derive(Debug, Default)]
pub struct StreamingCostTracker {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub cached_tokens: u64,
    pub cache_creation_tokens: u64,
    /// Image tokens for vision/multimodal requests
    pub image_tokens: u64,
    /// Number of images in the request
    pub image_count: u64,
    /// Provider-reported cost in USD (e.g., from Perplexity's cost breakdown)
    pub provider_reported_cost_usd: Option<f64>,
    /// Whether we've received a usage-only chunk (OpenRouter post-finish)
    pub received_usage_only_chunk: bool,
}

impl StreamingCostTracker {
    pub fn new() -> Self {
        Self {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            cached_tokens: 0,
            cache_creation_tokens: 0,
            image_tokens: 0,
            image_count: 0,
            provider_reported_cost_usd: None,
            received_usage_only_chunk: false,
        }
    }

    /// Extract usage from a chunk if present. Handles multiple provider formats:
    /// - OpenAI: usage in final chunk when stream_options.include_usage is true
    /// - OpenRouter: usage-only chunks after stream completion
    /// - Perplexity: provider-reported cost in usage.cost field
    pub fn accumulate(&mut self, chunk: &Value) {
        if let Some(usage) = chunk.get("usage") {
            // Track if this is a usage-only chunk (no choices or empty choices)
            let is_usage_only = chunk
                .get("choices")
                .map(|c| c.as_array().map(|a| a.is_empty()).unwrap_or(true))
                .unwrap_or(true);

            if is_usage_only {
                self.received_usage_only_chunk = true;
            }

            if let Some(prompt) = usage.get("prompt_tokens").and_then(Value::as_u64) {
                self.prompt_tokens = prompt;
            }
            if let Some(completion) = usage.get("completion_tokens").and_then(Value::as_u64) {
                self.completion_tokens = completion;
            }
            if let Some(total) = usage.get("total_tokens").and_then(Value::as_u64) {
                self.total_tokens = total;
            }

            // Handle prompt_tokens_details (OpenAI format)
            if let Some(details) = usage.get("prompt_tokens_details") {
                if let Some(cached) = details.get("cached_tokens").and_then(Value::as_u64) {
                    self.cached_tokens = cached;
                }
                if let Some(creation) = details.get("cache_creation_tokens").and_then(Value::as_u64)
                {
                    self.cache_creation_tokens = creation;
                }
                // Handle image tokens if present in details
                if let Some(image_tokens) = details.get("image_tokens").and_then(Value::as_u64) {
                    self.image_tokens = image_tokens;
                }
            }

            // Handle provider-reported cost (Perplexity format)
            // Perplexity sends cost in usage.cost as a number or breakdown object
            if let Some(cost) = usage.get("cost")
                && let Some(cost_value) = Self::extract_cost_value(cost)
            {
                self.provider_reported_cost_usd = Some(cost_value);
            }

            // Also check for cost in completion_tokens_details (some providers)
            if let Some(completion_details) = usage.get("completion_tokens_details")
                && let Some(cost) = completion_details.get("cost")
                && let Some(cost_value) = Self::extract_cost_value(cost)
            {
                self.provider_reported_cost_usd = Some(cost_value);
            }
        }
    }

    /// Count images in message content for vision/multimodal requests.
    /// Supports OpenAI format (image_url), Anthropic format (image source), and base64 images.
    pub fn count_images_in_messages(messages: &Value) -> (u64, u64) {
        let mut image_count = 0u64;
        let mut estimated_image_tokens = 0u64;

        if let Some(messages_array) = messages.as_array() {
            for message in messages_array {
                if let Some(content) = message.get("content") {
                    // Handle string content (no images)
                    if content.is_string() {
                        continue;
                    }

                    // Handle array content (multimodal)
                    if let Some(content_array) = content.as_array() {
                        for part in content_array {
                            if let Some(part_type) = part.get("type").and_then(Value::as_str) {
                                match part_type {
                                    "image_url" => {
                                        image_count += 1;
                                        // Estimate tokens based on image detail
                                        // OpenAI: low=85, high=1105, auto=depends on size
                                        if let Some(image_url) = part.get("image_url") {
                                            if let Some(detail) =
                                                image_url.get("detail").and_then(Value::as_str)
                                            {
                                                match detail {
                                                    "low" => estimated_image_tokens += 85,
                                                    "high" => estimated_image_tokens += 1105,
                                                    _ => estimated_image_tokens += 850, // auto/default
                                                }
                                            } else {
                                                estimated_image_tokens += 850; // default
                                            }
                                        } else {
                                            estimated_image_tokens += 850; // default
                                        }
                                    }
                                    "image" => {
                                        // Anthropic format
                                        image_count += 1;
                                        // Anthropic images are typically ~1500 tokens
                                        estimated_image_tokens += 1500;
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
        }

        (image_count, estimated_image_tokens)
    }

    /// Extract cost value from various formats:
    /// - Direct number: 0.001
    /// - Breakdown object: {"total_cost": 0.001}
    /// - Nested breakdown: {"cost": {"total_cost": 0.001}}
    fn extract_cost_value(cost: &Value) -> Option<f64> {
        // Direct number
        if let Some(cost_num) = cost.as_f64() {
            return Some(cost_num);
        }

        // Breakdown object with total_cost
        if let Some(total_cost) = cost.get("total_cost").and_then(Value::as_f64) {
            return Some(total_cost);
        }

        // Nested cost object
        if let Some(nested_cost) = cost.get("cost") {
            return Self::extract_cost_value(nested_cost);
        }

        None
    }

    /// Check if we have valid usage data
    pub fn has_usage(&self) -> bool {
        self.prompt_tokens > 0 || self.completion_tokens > 0
    }

    /// Get the effective cost: provider-reported cost if available, otherwise None
    /// (caller should calculate from token counts if not available)
    pub fn get_effective_cost_usd(&self) -> Option<f64> {
        self.provider_reported_cost_usd
    }
}

/// Extracts and preserves provider-specific fields from chunks.
pub fn extract_provider_specific_fields(chunk: &Value) -> Option<Map<String, Value>> {
    // Look for provider_specific_fields in the chunk
    if let Some(fields) = chunk
        .get("provider_specific_fields")
        .and_then(Value::as_object)
    {
        return Some(fields.clone());
    }

    // Also check in choices[0]
    if let Some(choices) = chunk.get("choices").and_then(Value::as_array)
        && let Some(first_choice) = choices.first()
        && let Some(fields) = first_choice
            .get("provider_specific_fields")
            .and_then(Value::as_object)
    {
        return Some(fields.clone());
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_tool_call_accumulator_basic() {
        let mut accumulator = ToolCallAccumulator::new();

        // First chunk with tool call start
        let chunk1 = json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\":"
                        }
                    }]
                }
            }]
        });

        accumulator
            .accumulate_tool_call_delta(chunk1.get("choices").unwrap()[0].get("delta").unwrap());

        // Second chunk with continued arguments
        let chunk2 = json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {
                            "arguments": "\"New York\"}"
                        }
                    }]
                }
            }]
        });

        accumulator
            .accumulate_tool_call_delta(chunk2.get("choices").unwrap()[0].get("delta").unwrap());

        assert!(accumulator.has_tool_calls());
        let result = accumulator.to_json().unwrap();
        let tool_calls = result.as_array().unwrap();
        assert_eq!(tool_calls.len(), 1);

        let tool_call = &tool_calls[0];
        assert_eq!(tool_call["id"], "call_abc123");
        assert_eq!(tool_call["type"], "function");
        assert_eq!(tool_call["function"]["name"], "get_weather");
        assert_eq!(
            tool_call["function"]["arguments"],
            "{\"location\":\"New York\"}"
        );
    }

    #[test]
    fn test_tool_call_accumulator_none_arguments() {
        let mut accumulator = ToolCallAccumulator::new();

        // Azure/Mistral may send null arguments
        let chunk = json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_xyz",
                        "function": {
                            "name": "test",
                            "arguments": null
                        }
                    }]
                }
            }]
        });

        accumulator
            .accumulate_tool_call_delta(chunk.get("choices").unwrap()[0].get("delta").unwrap());

        assert!(accumulator.has_tool_calls());
        let result = accumulator.to_json().unwrap();
        let tool_calls = result.as_array().unwrap();
        assert_eq!(tool_calls[0]["function"]["arguments"], "");
    }

    #[test]
    fn test_finish_reason_tracker() {
        let mut tracker = FinishReasonTracker::new();

        // Chunk with content and finish_reason
        let chunk1 = json!({
            "choices": [{
                "delta": {
                    "content": "Hello"
                },
                "finish_reason": "stop"
            }]
        });

        let finish = tracker.process_chunk(&chunk1);
        // Should not emit finish reason yet because delta has content
        assert!(finish.is_none());

        // Trailing empty delta chunk
        let chunk2 = json!({
            "choices": [{
                "delta": {},
                "finish_reason": "stop"
            }]
        });

        let finish = tracker.process_chunk(&chunk2);
        // Should emit finish reason now
        assert_eq!(finish, Some("stop".to_string()));
        assert!(tracker.has_emitted_finish_reason());
    }

    #[test]
    fn test_thinking_block_tracker() {
        let mut tracker = ThinkingBlockTracker::new();

        let chunk = json!({
            "choices": [{
                "delta": {
                    "reasoning_content": "Let me think about this..."
                }
            }]
        });

        tracker.process_chunk(&chunk);

        assert!(tracker.has_sent_first_thinking_block());
        assert_eq!(tracker.get_thinking_content(), "Let me think about this...");
    }
}
