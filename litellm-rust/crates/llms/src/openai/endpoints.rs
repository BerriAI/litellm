use litellm_core_utils::{
    ApiUrlError,
    url_utils::{ApiUrl, Complete, EndpointTarget, Mount},
};

pub const DEFAULT_API_BASE: &str = "https://api.openai.com/v1";

crate::provider_endpoints! { pub enum OpenAiEndpoint { Responses => POST "/responses", } }

pub fn legacy_responses_target(value: &str) -> Result<EndpointTarget, ApiUrlError> {
    let url = ApiUrl::<Complete>::parse_exact(value)?;
    if OpenAiEndpoint::Responses
        .path()
        .is_suffix_of(url.as_url().path())
    {
        return Ok(EndpointTarget::Exact(url));
    }
    Ok(EndpointTarget::Mount(ApiUrl::<Mount>::parse_mount(value)?))
}

pub fn resolve_websocket(
    value: &str,
    model: &str,
) -> Result<litellm_core_utils::url_utils::WebSocketUrl<Complete>, ApiUrlError> {
    use litellm_core_utils::url_utils::{QueryParameter, QueryPolicy, WebSocketUrl};
    let base = if value.trim().is_empty() {
        DEFAULT_API_BASE
    } else {
        value.trim()
    };
    let mut url = reqwest::Url::parse(base)?;
    match url.scheme() {
        "https" => {
            url.set_scheme("wss")
                .map_err(|()| ApiUrlError::InvalidTransport)?;
        }
        "http" => {
            url.set_scheme("ws")
                .map_err(|()| ApiUrlError::InvalidTransport)?;
        }
        "ws" | "wss" => {}
        _ => return Err(ApiUrlError::InvalidTransport),
    }
    let complete = OpenAiEndpoint::Responses.path().is_suffix_of(url.path());
    let segments = OpenAiEndpoint::Responses
        .path()
        .segments()
        .filter(|_| !complete);
    WebSocketUrl::<Mount>::parse_mount(url.as_str())?.resolve_segments(
        segments,
        &[QueryParameter {
            key: "model",
            value: model,
            policy: QueryPolicy::Default,
        }],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_llm::endpoint::ProviderEndpoint;
    use rstest::rstest;

    #[rstest]
    #[case::root("https://gateway.test", "/responses")]
    #[case::v1("https://gateway.test/v1", "/v1/responses")]
    #[case::v2("https://gateway.test/v2", "/v2/responses")]
    #[case::complete("https://gateway.test/team/responses", "/team/responses")]
    fn http_legacy_policy_places_route_before_query(#[case] base: &str, #[case] path: &str) {
        let value = format!("{base}?tenant=a%2fb&tenant=c");
        let endpoint = OpenAiEndpoint::Responses
            .resolve(&legacy_responses_target(&value).unwrap())
            .unwrap();
        assert_eq!(endpoint.url().as_url().path(), path);
        assert_eq!(
            endpoint.url().as_url().query(),
            Some("tenant=a%2fb&tenant=c")
        );
    }

    #[rstest]
    #[case::root("https://gateway.test/v2", "/v2/responses")]
    #[case::complete("wss://gateway.test/v2/responses", "/v2/responses")]
    fn websocket_completion_selects_transport_before_resolving(
        #[case] base: &str,
        #[case] path: &str,
    ) {
        let url = resolve_websocket(base, "a/b %é").unwrap();
        assert_eq!(url.as_url().scheme(), "wss");
        assert_eq!(url.as_url().path(), path);
        assert_eq!(
            url.as_url()
                .query_pairs()
                .find(|(key, _)| key == "model")
                .unwrap()
                .1,
            "a/b %é"
        );
    }

    #[rstest]
    fn websocket_preserves_encoded_existing_model_without_duplicate() {
        let url = resolve_websocket(
            "https://gateway.test/v2/responses?%6dodel=existing&sig=a%2fb+%20",
            "ignored",
        )
        .unwrap();
        assert_eq!(url.as_url().query(), Some("%6dodel=existing&sig=a%2fb+%20"));
        assert_eq!(
            url.as_url()
                .query_pairs()
                .filter(|(key, _)| key == "model")
                .count(),
            1
        );
    }
}
