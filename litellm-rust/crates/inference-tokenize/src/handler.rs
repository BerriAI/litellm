use std::time::Duration;

use litellm_host::{
    interceptors::{
        Cost, ExecutionFacts, Interceptors, ProviderIdentity, RequestContext, ResultSource,
        WireRequest,
    },
    observation::ObservationSender,
};
use litellm_http::transport::Error as TransportError;
use litellm_inference::{context::CallContext, provider::ResolvedProvider};
use litellm_llms::base_llm::{
    auth::{AuthScheme, resolve_auth},
    tokenize::transformation::{BaseTokenizeConfig, TokenizeContext},
};
use litellm_llms_types::formats::tokenize::{TokenizeOutput, TokenizeResponse};
use litellm_secrets::source::Secrets;
use serde_json::{Map, Value};

use crate::{Error, TokenizeRequest, TokenizeRoute};

pub(crate) async fn execute(
    route: &TokenizeRoute,
    call: &TokenizeRequest,
    config: &dyn BaseTokenizeConfig,
    provider: ResolvedProvider<'_>,
    secrets: Secrets,
    interceptors: &impl Interceptors<Error>,
    observers: Option<&ObservationSender>,
) -> Result<TokenizeResponse, Error> {
    let context = CallContext::new(interceptors, observers.cloned().into());
    let env_lookup = |key: &str| secrets.get(key);
    let input = TokenizeContext::new(
        call.api_key.as_ref(),
        call.api_base.as_ref(),
        &call.extra_headers,
        &call.deployment_params,
        &call.optional_params,
    );
    config.validate_tokenize_input(&input, &env_lookup)?;
    let supported_params = config.get_supported_tokenize_params();
    let mut params: Map<String, Value> = call
        .deployment_params
        .iter()
        .filter(|(key, _)| supported_params.contains(&key.as_str()))
        .chain(call.optional_params.iter())
        .map(|(key, value)| (key.clone(), value.value().clone()))
        .collect();
    if let Some(value) = call.add_special_tokens {
        params.insert("add_special_tokens".into(), Value::Bool(value));
    }
    let params = config.map_tokenize_params(provider.model, &params)?;
    let environment = config.validate_environment(&input, &params, &env_lookup)?;
    let url = config.get_complete_url(
        call.api_base.as_ref().map(|base| base.value().as_str()),
        provider.model,
        &params,
        &env_lookup,
    )?;
    let body =
        config.transform_tokenize_request(provider.model, &call.prompt, &params, &env_lookup)?;
    let iam_config = match &environment.auth {
        AuthScheme::WatsonxIam { config } => Some(config.clone()),
        _ => None,
    };
    let mut authenticated =
        resolve_auth(&route.auth, &route.http, environment, &env_lookup).await?;
    let wire = interceptors
        .before_provider_request(
            WireRequest {
                url,
                headers: authenticated.headers,
                body,
            },
            RequestContext {
                model: provider.model.into(),
                custom_llm_provider: <&str>::from(provider.provider).into(),
                optional_params: Value::Object(params),
                secret_fields: config
                    .secret_param_names()
                    .iter()
                    .map(|name| (*name).into())
                    .collect(),
                api_key: call.api_key.as_ref().map(|api_key| api_key.value().clone()),
            },
        )
        .await?;
    authenticated.headers = wire.headers;
    let bearer = authenticated
        .headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
        .and_then(|(_, value)| value.strip_prefix("Bearer "))
        .map(litellm_auth::SecretValue::new);
    let outbound = litellm_inference::outbound::outbound_request(
        authenticated,
        wire.url,
        &wire.body,
        Some(call.timeout.unwrap_or(Duration::from_secs(60))),
    )?;
    let response = litellm_inference::outbound::send(outbound, &route.http)
        .await
        .map_err(TransportError::from_reqwest_before_dispatch)?;
    let status = response.status();
    if let (401, Some(config), Some(bearer)) =
        (status.as_u16(), iam_config.as_ref(), bearer.as_ref())
    {
        route
            .auth
            .watsonx
            .invalidate_if_matches(config, bearer)
            .await;
    }
    let text = response.text().await.map_err(TransportError::from)?;
    if !status.is_success() {
        return Err(TransportError::Http {
            status: status.as_u16(),
            body: litellm_http::request::truncate_error_body(&text),
        }
        .into());
    }
    context.response_received(&text).await?;
    let json = serde_json::from_str(&text).map_err(|error| {
        Error::InvalidResponse(litellm_llms::ErrorDetail::invalid(
            "tokenize response JSON",
            error,
        ))
    })?;
    let response = project_response(
        config.transform_tokenize_response(provider.model, json)?,
        call.include.output(),
    )?;
    context
        .result_ready(ExecutionFacts {
            provider: ProviderIdentity {
                model: provider.model.into(),
                provider: <&str>::from(provider.provider).into(),
            },
            source: ResultSource::Provider,
            cost: Cost::Deferred,
        })
        .await?;
    Ok(response)
}

fn project_response(
    response: TokenizeResponse,
    output: TokenizeOutput,
) -> Result<TokenizeResponse, Error> {
    let require_ids = output.requires_ids();
    let require_tokens = output.requires_tokens();
    if require_ids && response.token_ids.is_none() {
        return Err(Error::InvalidResponse(
            litellm_llms::ErrorDetail::InvalidValue {
                field: "token_ids",
                expected: "a provider token ID representation",
                actual: Value::Null,
            },
        ));
    }
    if require_tokens && response.tokens.is_none() {
        return Err(Error::InvalidResponse(
            litellm_llms::ErrorDetail::InvalidValue {
                field: "tokens",
                expected: "a provider token piece representation",
                actual: Value::Null,
            },
        ));
    }
    if output == TokenizeOutput::Available {
        return Ok(response);
    }
    Ok(TokenizeResponse {
        model: response.model,
        count: response.count,
        token_ids: require_ids.then_some(response.token_ids).flatten(),
        tokens: require_tokens.then_some(response.tokens).flatten(),
    })
}

#[cfg(test)]
mod tests {
    use litellm_llms_types::formats::tokenize::{TokenizeOutput, TokenizeResponse};
    use rstest::rstest;

    use super::project_response;
    use crate::Error;

    fn response() -> TokenizeResponse {
        TokenizeResponse {
            model: "model".into(),
            count: 0,
            token_ids: Some(Vec::new()),
            tokens: Some(Vec::new()),
        }
    }

    #[rstest]
    #[case::count(TokenizeOutput::Count, None, None)]
    #[case::ids(TokenizeOutput::Ids, Some(Vec::new()), None)]
    #[case::tokens(TokenizeOutput::Tokens, None, Some(Vec::new()))]
    #[case::both(TokenizeOutput::Both, Some(Vec::new()), Some(Vec::new()))]
    fn strict_outputs_preserve_empty_arrays_and_project_other_fields(
        #[case] output: TokenizeOutput,
        #[case] ids: Option<Vec<u64>>,
        #[case] tokens: Option<Vec<String>>,
    ) {
        let projected = project_response(response(), output).unwrap();

        assert_eq!(projected.count, 0);
        assert_eq!(projected.token_ids, ids);
        assert_eq!(projected.tokens, tokens);
    }

    #[rstest]
    #[case::ids(TokenizeOutput::Ids)]
    #[case::tokens(TokenizeOutput::Tokens)]
    #[case::both(TokenizeOutput::Both)]
    fn strict_outputs_reject_missing_representations(#[case] output: TokenizeOutput) {
        let mut provider_response = response();
        provider_response.token_ids = None;
        provider_response.tokens = None;

        assert!(matches!(
            project_response(provider_response, output),
            Err(Error::InvalidResponse(_))
        ));
    }
}
