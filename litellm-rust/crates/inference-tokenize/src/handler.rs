use std::time::Duration;

use litellm_host::{
    interceptors::{
        Cost, ExecutionFacts, Interceptors, ProviderIdentity, RawResponse, RequestContext,
        ResultSource, WireRequest,
    },
    lifecycle::{CallEvent, ExecutionEvent},
    observation::ObservationSender,
};
use litellm_http::{request::string_headers, transport::Error as TransportError};
use litellm_inference::provider::ResolvedProvider;
use litellm_llms::base_llm::{auth::resolve_auth, tokenize::transformation::BaseTokenizeConfig};
use litellm_llms_types::formats::tokenize::TokenizeResponse;
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
    let env_lookup = |key: &str| secrets.get(key);
    let supported_params = config.get_supported_tokenize_params();
    let params: Map<String, Value> = call
        .deployment_params
        .iter()
        .filter(|(key, _)| supported_params.contains(&key.as_str()))
        .chain(call.optional_params.iter())
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let params = config.map_tokenize_params(provider.model, &params)?;
    let headers = string_headers("tokenize", Some(call.extra_headers.clone()))?;
    let environment = config.validate_environment(
        headers,
        call.api_key.as_ref().map(|key| key.expose()),
        &params,
        &env_lookup,
    )?;
    let url = config.get_complete_url(
        call.api_base.as_deref(),
        provider.model,
        &params,
        &env_lookup,
    )?;
    let body =
        config.transform_tokenize_request(provider.model, &call.prompt, &params, &env_lookup)?;
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
                api_key: call.api_key.clone(),
            },
        )
        .await?;
    authenticated.headers = wire.headers;
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
    let text = response.text().await.map_err(TransportError::from)?;
    if !status.is_success() {
        return Err(TransportError::Http {
            status: status.as_u16(),
            body: litellm_http::request::truncate_error_body(&text),
        }
        .into());
    }
    let raw = RawResponse { body: text };
    if let Some(observers) = observers {
        observers.emit(CallEvent::Execution(
            ExecutionEvent::ProviderResponseReceived { raw: raw.clone() },
        ));
    }
    interceptors
        .after_provider_response(raw.clone())
        .await
        .map_err(Error::post_call)?;
    let json = serde_json::from_str(&raw.body).map_err(|error| {
        Error::InvalidResponse(litellm_llms::ErrorDetail::invalid(
            "tokenize response JSON",
            error,
        ))
    })?;
    let response = config.transform_tokenize_response(provider.model, json)?;
    interceptors
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
