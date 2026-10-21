use std::time::Duration;

use litellm_inference_messages::MessagesShaping;
use litellm_router_types::LitellmParams;

use crate::{Error, policy::seconds};

#[derive(Clone, Debug, Default)]
pub struct Deployment {
    pub model: String,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub litellm_params: LitellmParams,
    pub timeout: Option<Duration>,
    pub stream_timeout: Option<Duration>,
    pub shaping: MessagesShaping,
    /// `model_info.team_id`: the team that owns this deployment, which only that team's keys
    /// may use. `None` for a deployment every caller shares.
    pub team_id: Option<String>,
    /// `model_info.id`, which names the deployment in state shared with other replicas and
    /// with Python routers on the same Redis.
    pub id: Option<String>,
    /// The model a deployment serves under another name, from `litellm_params.base_model` or
    /// `model_info.base_model`, which Azure's param rules and prices follow.
    pub base_model: Option<String>,
    /// `model_info.discoverable: false`: the deployment still serves requests, but only the
    /// master key sees it in model listings.
    pub undiscoverable: bool,
}

impl Deployment {
    pub fn from_params(params: &LitellmParams) -> Result<Self, Error> {
        let timeout = |setting, value: &Option<_>| {
            value
                .as_ref()
                .map(|value| seconds(setting, value))
                .transpose()
                .map(Option::flatten)
        };
        Ok(Self {
            model: params.model.clone(),
            api_key: params
                .api_key
                .as_ref()
                .map(|value| value.expose().to_string()),
            api_base: params.api_base.clone(),
            custom_llm_provider: params.custom_llm_provider.clone(),
            timeout: timeout("litellm_params.timeout", &params.timeout)?,
            stream_timeout: timeout("litellm_params.stream_timeout", &params.stream_timeout)?,
            litellm_params: params.clone(),
            shaping: MessagesShaping::default(),
            team_id: None,
            id: None,
            undiscoverable: false,
            base_model: params
                .extra
                .get("base_model")
                .and_then(|value| value.as_str())
                .filter(|name| !name.is_empty())
                .map(str::to_owned),
        })
    }

    pub fn timeout_for(&self, stream: bool) -> Option<Duration> {
        match stream {
            true => self.stream_timeout.or(self.timeout),
            false => self.timeout,
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::unary_uses_timeout(Some(30), Some(5), false, Some(30))]
    #[case::stream_uses_stream_timeout(Some(30), Some(5), true, Some(5))]
    #[case::stream_falls_back_to_timeout(Some(30), None, true, Some(30))]
    #[case::unary_ignores_stream_timeout(None, Some(5), false, None)]
    fn timeout_follows_the_request_mode(
        #[case] timeout: Option<u64>,
        #[case] stream_timeout: Option<u64>,
        #[case] stream: bool,
        #[case] expected: Option<u64>,
    ) {
        let deployment = Deployment {
            timeout: timeout.map(Duration::from_secs),
            stream_timeout: stream_timeout.map(Duration::from_secs),
            ..Deployment::default()
        };

        assert_eq!(
            deployment.timeout_for(stream),
            expected.map(Duration::from_secs)
        );
    }
}
