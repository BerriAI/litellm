use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CooldownDeployment {
    pub model_id: String,
    pub api_base: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct ErrorInformation {
    pub error_code: Option<String>,
    pub error_class: Option<String>,
    pub llm_provider: Option<String>,
    pub traceback: Option<String>,
    pub error_message: Option<String>,
    pub error_provider_request_id: Option<String>,
    pub error_rate_limit_category: Option<String>,
    pub error_rate_limit_type: Option<String>,
    pub error_budget_entity_type: Option<String>,
    pub error_budget_entity_id: Option<String>,
    pub error_budget_limit: Option<f64>,
    pub error_budget_spend: Option<f64>,
    pub error_cooldown_deployments: Option<Vec<CooldownDeployment>>,
    pub normalized_error: Option<String>,
}
