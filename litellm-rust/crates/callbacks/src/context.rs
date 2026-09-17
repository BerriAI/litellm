#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallContext {
    pub call_type: String,
    pub model: String,
    pub custom_llm_provider: String,
    pub litellm_call_id: String,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CallTiming {
    pub start_time: f64,
    pub end_time: f64,
}
