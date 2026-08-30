//! Streaming chunk processing enhancements.
//!
//! Provides enhanced streaming chunk processing including:
//! - Function/tool call parsing and accumulation
//! - Thinking block handling
//! - Finish reason tracking
//! - Provider-specific field preservation

use serde_json::{Map, Value};

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
                let index = tool_call
                    .get("index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize;

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
                    if let Some(args) = function.get("arguments") {
                        if let Some(args_str) = args.as_str() {
                            state.arguments.push_str(args_str);
                        }
                        // If arguments is null, we just skip it (Azure/Mistral behavior)
                    }
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
            if let Some(args) = function_call.get("arguments") {
                if let Some(args_str) = args.as_str() {
                    state.arguments.push_str(args_str);
                }
            }
        }
    }

    /// Check if we have any accumulated tool calls
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
            && self.tool_calls.iter().any(|tc| tc.name.is_some() || !tc.arguments.is_empty())
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
                function.insert(
                    "arguments".to_string(),
                    Value::String(tc.arguments.clone()),
                );
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
        if let Some(choices) = chunk.get("choices").and_then(Value::as_array) {
            if let Some(first_choice) = choices.first() {
                // Check for finish_reason in the choice
                if let Some(finish_reason) = first_choice.get("finish_reason").and_then(Value::as_str) {
                    self.intermittent_finish_reason = Some(finish_reason.to_string());
                    
                    // Check if delta is empty (trailing chunk)
                    let delta = first_choice.get("delta");
                    let is_empty_delta = delta
                        .map(|d| {
                            let has_content = d.get("content").and_then(Value::as_str).map(|s| !s.is_empty()).unwrap_or(false);
                            let has_tool_calls = d.get("tool_calls").and_then(Value::as_array).map(|a| !a.is_empty()).unwrap_or(false);
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
        if let Some(choices) = chunk.get("choices").and_then(Value::as_array) {
            if let Some(first_choice) = choices.first() {
                if let Some(delta) = first_choice.get("delta") {
                    // Check for thinking_blocks
                    if let Some(thinking_blocks) = delta.get("thinking_blocks").and_then(Value::as_array) {
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
            let is_usage_only = chunk.get("choices").map(|c| {
                c.as_array().map(|a| a.is_empty()).unwrap_or(true)
            }).unwrap_or(true);
            
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
                if let Some(creation) = details.get("cache_creation_tokens").and_then(Value::as_u64) {
                    self.cache_creation_tokens = creation;
                }
            }
            
            // Handle provider-reported cost (Perplexity format)
            // Perplexity sends cost in usage.cost as a number or breakdown object
            if let Some(cost) = usage.get("cost") {
                if let Some(cost_value) = Self::extract_cost_value(cost) {
                    self.provider_reported_cost_usd = Some(cost_value);
                }
            }
            
            // Also check for cost in completion_tokens_details (some providers)
            if let Some(completion_details) = usage.get("completion_tokens_details") {
                if let Some(cost) = completion_details.get("cost") {
                    if let Some(cost_value) = Self::extract_cost_value(cost) {
                        self.provider_reported_cost_usd = Some(cost_value);
                    }
                }
            }
        }
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
    if let Some(fields) = chunk.get("provider_specific_fields").and_then(Value::as_object) {
        return Some(fields.clone());
    }

    // Also check in choices[0]
    if let Some(choices) = chunk.get("choices").and_then(Value::as_array) {
        if let Some(first_choice) = choices.first() {
            if let Some(fields) = first_choice.get("provider_specific_fields").and_then(Value::as_object) {
                return Some(fields.clone());
            }
        }
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

        accumulator.accumulate_tool_call_delta(chunk1.get("choices").unwrap()[0].get("delta").unwrap());

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

        accumulator.accumulate_tool_call_delta(chunk2.get("choices").unwrap()[0].get("delta").unwrap());

        assert!(accumulator.has_tool_calls());
        let result = accumulator.to_json().unwrap();
        let tool_calls = result.as_array().unwrap();
        assert_eq!(tool_calls.len(), 1);
        
        let tool_call = &tool_calls[0];
        assert_eq!(tool_call["id"], "call_abc123");
        assert_eq!(tool_call["type"], "function");
        assert_eq!(tool_call["function"]["name"], "get_weather");
        assert_eq!(tool_call["function"]["arguments"], "{\"location\":\"New York\"}");
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

        accumulator.accumulate_tool_call_delta(chunk.get("choices").unwrap()[0].get("delta").unwrap());

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
