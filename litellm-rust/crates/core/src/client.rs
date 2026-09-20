use std::sync::Arc;

use litellm_cache::LLMCachingHandler;
use litellm_core_utils::{
    budget::Budget, cost_calculator::messages_cost, token_counter::get_modified_max_tokens,
};
use litellm_token_counter::{CountableRequest, TokenCounter};
use serde_json::{Map, Value};

#[derive(Clone, Default)]
pub struct ClientOptions {
    pub cache: Option<LLMCachingHandler>,
    pub budget: Option<Arc<Budget>>,
    pub model_info: Option<Value>,
    pub token_counter: Option<Arc<TokenCounter>>,
    pub modify_params: bool,
}

impl ClientOptions {
    pub fn adjust_max_tokens(&self, body: &mut Map<String, Value>) {
        if !self.modify_params {
            return;
        }
        let (Some(counter), Some(info), Some(requested)) = (
            &self.token_counter,
            &self.model_info,
            body.get("max_tokens").and_then(Value::as_i64),
        ) else {
            return;
        };
        let request =
            serde_json::json!({"model": body.get("model"), "messages": body.get("messages")});
        let Ok(request) = serde_json::from_value::<CountableRequest>(request) else {
            return;
        };
        let Ok(count) = counter.count_request(&request) else {
            return;
        };
        let output = info
            .get("max_output_tokens")
            .or_else(|| info.get("max_tokens"))
            .and_then(Value::as_i64);
        let adjusted = get_modified_max_tokens(
            requested,
            info.get("max_input_tokens").and_then(Value::as_i64),
            output,
            count.input_tokens,
            None,
            None,
        );
        body.insert("max_tokens".into(), Value::from(adjusted));
    }

    pub fn record_messages_usage(&self, usage: &Value, cache_hit: bool) -> Option<f64> {
        let cost = if cache_hit {
            Some(0.0)
        } else {
            self.model_info
                .as_ref()
                .and_then(|info| messages_cost(info, usage))
        };
        if !cache_hit && let (Some(budget), Some(cost)) = (&self.budget, cost) {
            budget.record(cost);
        }
        cost
    }
}
