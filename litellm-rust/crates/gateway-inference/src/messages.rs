//! `POST /v1/messages`, as the Python proxy's `anthropic_response` serves it.

use litellm_gateway_auth::AuthenticatedRequest;
use std::sync::Arc;

use axum::{
    Json,
    body::Bytes,
    extract::State,
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use litellm_host_http::Sse;
use litellm_inference_messages::{ChatCompletions, MessagesCall, messages_body, route::Messages};
use litellm_llms::anthropic::adapters::transformation::UpstreamReport;
use litellm_llms_types::headers::{ProviderSpecificHeader, ProviderSpecificHeaders};
use serde_json::{Map, Value};

use litellm_gateway_hooks::Route;

use crate::{Deployment, Error, Gateway, JsonBody, RequestId, request};

/// Client headers Python forwards to Anthropic-speaking providers on every call.
const ANTHROPIC_API_HEADERS: [&str; 2] = ["anthropic-version", "anthropic-beta"];
const OPENROUTER_API_HEADERS: [&str; 2] = [
    "x-openrouter-metadata",
    "x-openrouter-experimental-metadata",
];
const ANTHROPIC_API_HEADER_PROVIDERS: &str =
    "anthropic,bedrock,bedrock_mantle,vertex_ai,openrouter,deepseek";

pub async fn create(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    RequestId(request_id): RequestId,
    headers: HeaderMap,
    body: Result<JsonBody, Error>,
) -> impl IntoResponse {
    let result = match body {
        Ok(body) => handle(&gateway, &identity, &headers, body).await,
        Err(error) => Err(error),
    };
    result.map_err(|error| error.messages_response(request_id.as_deref()))
}

async fn handle(
    gateway: &Gateway,
    identity: &AuthenticatedRequest,
    headers: &HeaderMap,
    JsonBody { object, raw }: JsonBody,
) -> Result<Response, Error> {
    gateway.check_body(&object)?;
    let group = request::resolve_group(gateway, &object)?;
    let name = request::model_name(&object)?.to_owned();
    let tracking = crate::hooks::admit(
        gateway,
        identity,
        Route::Messages,
        headers,
        &raw,
        &object,
        &name,
    )
    .await?;
    let (mut body, cache_options) = match crate::caching::prepare(identity, object) {
        Ok(prepared) => prepared,
        Err(error) => {
            if let Some(tracking) = tracking {
                tracking.fail(&error);
            }
            return Err(error);
        }
    };
    let edit = |body: &mut Map<String, Value>| {
        if let Some(tracking) = &tracking {
            tracking.edit_body(body);
        }
    };
    edit(&mut body);
    let reparse = || -> Result<Map<String, Value>, Error> {
        let mut body = crate::caching::prepare(identity, request::object(&raw)?)?.0;
        edit(&mut body);
        Ok(body)
    };
    let chat = ChatCompletions {
        route: gateway.chat_completions.clone(),
        drop_params: gateway.drop_params(),
        model_info: gateway.model_info(),
    };
    let route = gateway.messages.clone();
    let route = match &gateway.cache {
        Some(cache) => route.with_cache(litellm_cache_response::ScopedCache::new(
            cache.clone(),
            cache_options.scope.clone(),
        )),
        None => route,
    };

    let upstream_headers = &upstream_headers(headers, tracking.as_ref());
    let cache_headers = crate::caching::CacheHeaders::default();
    let mut body = Some(body);
    let routed = gateway
        .models
        .route_where(
            group,
            request::usable(identity),
            |attempt| {
                // Every attempt hands its body to the provider. The first takes the parsed
                // body and a retry parses the request again, so no request clones a body
                // that can hold a million tokens.
                let body = body.take().map_or_else(&reparse, Ok);
                let route = route.clone();
                let chat = chat.clone();
                let cache_headers = cache_headers.clone();
                let name = name.as_str();
                if let Some(tracking) = &tracking {
                    tracking.attempt(name, &attempt);
                }
                async move {
                    crate::observe::record(name, attempt.deployment);
                    request::authorize(identity, name, attempt.deployment).await?;
                    let call = project(attempt.deployment, body?, upstream_headers)?;
                    // A translated answer's usage and model reach the hooks through `report`,
                    // since the placeholder usage its stream opens with is not the provider's.
                    let report = ChatCompletions::serves(&call).then(UpstreamReport::default);
                    let machine = match &report {
                        Some(report) => chat.machine(call, report.clone()),
                        None => route.machine(call, cache_options.policy),
                    };
                    let stream = Sse::<Messages, _, _>::new(Json, |error| {
                        Bytes::from(Error::from(error).sse_frame())
                    });
                    let mut response =
                        litellm_host_http::serve(machine, (), cache_headers, stream, None).await?;
                    if let Some(report) = report {
                        response.extensions_mut().insert(report);
                    }
                    Ok(response)
                }
            },
            Error::failure,
        )
        .await
        .ok_or_else(|| Error::UnknownModel(name.clone()))
        .and_then(|routed| routed)
        .map(|routed| cache_headers.apply(request::hold(routed)));
    match tracking {
        Some(tracking) => tracking.finish(routed).await,
        None => routed,
    }
}

/// Client headers the provider gets: the API headers scoped to the providers that take them,
/// and any header a hook set.
#[derive(Default)]
struct UpstreamHeaders {
    scoped: Option<ProviderSpecificHeaders>,
    extra: Option<Map<String, Value>>,
}

fn upstream_headers(
    headers: &HeaderMap,
    tracking: Option<&crate::hooks::Tracking>,
) -> UpstreamHeaders {
    let client = |name: &str| Some(headers.get(name)?.to_str().ok()?.to_owned());
    let Some(tracking) = tracking else {
        return UpstreamHeaders {
            scoped: scoped_headers(client),
            extra: None,
        };
    };
    let mut names: Vec<&str> = header_scopes()
        .iter()
        .flat_map(|(_, names)| names.iter().copied())
        .collect();
    names.sort_unstable();
    names.dedup();
    let forwarded = names
        .iter()
        .filter_map(|name| Some(((*name).to_owned(), client(name)?)))
        .collect();
    let Some(sent) = tracking.edit_headers(forwarded) else {
        return UpstreamHeaders {
            scoped: scoped_headers(client),
            extra: None,
        };
    };
    let value = |name: &str| {
        sent.iter()
            .find(|(sent, _)| sent.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.clone())
    };
    let extra: Map<String, Value> = sent
        .iter()
        .filter(|(name, _)| !names.iter().any(|scoped| scoped.eq_ignore_ascii_case(name)))
        .map(|(name, value)| (name.clone(), Value::from(value.as_str())))
        .collect();
    UpstreamHeaders {
        scoped: scoped_headers(value),
        extra: (!extra.is_empty()).then_some(extra),
    }
}

fn project(
    deployment: &Deployment,
    body: Map<String, Value>,
    headers: &UpstreamHeaders,
) -> Result<MessagesCall, Error> {
    let body = body
        .into_iter()
        .map(|(name, value)| match name.as_str() {
            "model" => (name, Value::from(deployment.model.as_str())),
            _ => (name, value),
        })
        .collect();
    Ok(MessagesCall {
        body: messages_body(body)?,
        api_key: deployment.api_key.clone(),
        api_base: deployment.api_base.clone(),
        custom_llm_provider: deployment.custom_llm_provider.clone(),
        litellm_params: deployment.litellm_params.clone(),
        extra_headers: headers.extra.clone(),
        provider_specific_header: headers.scoped.clone(),
        timeout: deployment.timeout,
        shaping: deployment.shaping.clone(),
    })
}

fn header_scopes() -> [(&'static str, &'static [&'static str]); 2] {
    [
        (
            ANTHROPIC_API_HEADER_PROVIDERS,
            ANTHROPIC_API_HEADERS.as_slice(),
        ),
        ("openrouter", OPENROUTER_API_HEADERS.as_slice()),
    ]
}

fn scoped_headers(get: impl Fn(&str) -> Option<String>) -> Option<ProviderSpecificHeaders> {
    let forwarded: Vec<ProviderSpecificHeader> = header_scopes()
        .into_iter()
        .filter_map(|(provider, names)| {
            let extra_headers: Map<String, Value> = names
                .iter()
                .filter_map(|name| {
                    let value = get(name)?;
                    Some(((*name).to_owned(), Value::from(value)))
                })
                .collect();
            (!extra_headers.is_empty()).then(|| ProviderSpecificHeader {
                custom_llm_provider: provider.into(),
                extra_headers,
            })
        })
        .collect();
    (!forwarded.is_empty()).then_some(ProviderSpecificHeaders::Many(forwarded))
}
