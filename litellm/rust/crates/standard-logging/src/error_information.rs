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

/// The `error_information` the proxy stamps on a call whose client went away before it finished,
/// `_CLIENT_DISCONNECTED_ERROR_INFORMATION` of litellm/litellm_core_utils/litellm_logging.py, which
/// `get_error_information_for_logging_payload` logs in place of the exception's own.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClientDisconnected;

impl Serialize for ClientDisconnected {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;

        let mut fields = serializer.serialize_struct("ClientDisconnected", 3)?;
        fields.serialize_field("error_code", "499")?;
        fields.serialize_field("error_message", "Client disconnected the request")?;
        fields.serialize_field("error_class", "ClientDisconnected")?;
        fields.end()
    }
}

/// What a payload logs as `error_information`: the record read off the call's exception, or the
/// fixed three keys a client disconnect logs instead.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum LoggedErrorInformation {
    Raised(Box<ErrorInformation>),
    ClientDisconnected(ClientDisconnected),
}

impl From<ErrorInformation> for LoggedErrorInformation {
    fn from(information: ErrorInformation) -> Self {
        Self::Raised(Box::new(information))
    }
}

impl From<ClientDisconnected> for LoggedErrorInformation {
    fn from(disconnected: ClientDisconnected) -> Self {
        Self::ClientDisconnected(disconnected)
    }
}
