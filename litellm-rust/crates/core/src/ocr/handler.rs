use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use reqwest::{Method, Url};
use serde_json::Value;

use crate::Error;
use crate::auth::{AuthOperation, AuthRuntime, AuthSession, ReplayPolicy};
use crate::constants::OCR_AUTH_REQUEST_TIMEOUT_SECS;
use crate::provider_callbacks::ProviderAttemptObserver;
use crate::provider_callbacks::handler::{
    AuthenticatedProviderRequest, ProviderAttemptContext, ProviderHttpResponse, ProviderRequest,
    ProviderRequestBody, send_authenticated_provider_request,
};
use crate::request_context::LiteLlmRequestContext;

use super::prepare::OcrPlan;
use super::transformation::OcrResponseHandling;
use super::types::OcrResponseData;

#[tracing::instrument(
    name = "ocr",
    target = "litellm::function_trace",
    level = "trace",
    skip_all
)]
pub async fn ocr<Observer>(
    plan: OcrPlan,
    context: &LiteLlmRequestContext,
    runtime: Arc<AuthRuntime>,
    observer: &mut Observer,
) -> Result<OcrResponseData, Error>
where
    Observer: ProviderAttemptObserver,
    Observer::Error: std::fmt::Display,
{
    let context = LiteLlmRequestContext {
        litellm_call_id: Some(
            context
                .litellm_call_id
                .clone()
                .unwrap_or_else(|| format!("{:032x}", rand::random::<u128>())),
        ),
        ..context.clone()
    };
    let timeout = plan
        .timeout
        .unwrap_or(Duration::from_secs(OCR_AUTH_REQUEST_TIMEOUT_SECS));
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or_else(|| Error::InvalidRequest("OCR timeout is too large".into()))?;
    if timeout.is_zero() {
        return Err(Error::Network("Request timed out".into()));
    }
    let binding = tokio::time::timeout_at(
        deadline.into(),
        runtime
            .credentials
            .resolve(plan.auth.credential.clone(), runtime.clone()),
    )
    .await
    .map_err(|_| Error::Network("Request timed out".into()))??;
    let session = binding.bind(
        &plan.url,
        if plan.config.response_handling() == OcrResponseHandling::AzureDocumentIntelligencePoll {
            ReplayPolicy::SameOrigin
        } else {
            ReplayPolicy::Never
        },
    )?;
    let document = serde_json::to_value(&plan.document)
        .map_err(|_| Error::InvalidRequest("invalid OCR document".into()))?;
    let body = plan
        .config
        .transform_ocr_request(&plan.model, document, plan.optional_params.clone())?
        .data;
    let body = serde_json::from_value(body)
        .map_err(|_| Error::InvalidRequest("OCR provider body must be an object".into()))?;
    let response = send(
        &plan,
        &context,
        &runtime,
        &session,
        observer,
        Attempt {
            url: plan.url.clone(),
            body,
            number: 1,
            deadline,
        },
    )
    .await?;
    let response_json = if plan.config.response_handling()
        == OcrResponseHandling::AzureDocumentIntelligencePoll
        && response.status.as_u16() == 202
    {
        poll(
            &plan, &context, &runtime, &session, observer, response, deadline,
        )
        .await?
    } else {
        serde_json::from_str(&response.body)
            .map_err(|_| Error::InvalidResponse("invalid OCR response JSON".into()))?
    };
    let mut response = plan.config.transform_ocr_response_with_params(
        &plan.model,
        response_json,
        &plan.optional_params,
    )?;
    response.provider_native_response = None;
    Ok(response)
}

struct Attempt {
    url: Url,
    body: BTreeMap<String, Value>,
    number: u32,
    deadline: Instant,
}

async fn send<Observer>(
    plan: &OcrPlan,
    context: &LiteLlmRequestContext,
    runtime: &AuthRuntime,
    session: &AuthSession,
    observer: &mut Observer,
    attempt: Attempt,
) -> Result<ProviderHttpResponse, Error>
where
    Observer: ProviderAttemptObserver,
    Observer::Error: std::fmt::Display,
{
    let initial = attempt.number == 1;
    let request = runtime
        .http
        .request(
            if initial { Method::POST } else { Method::GET },
            attempt.url.clone(),
        )
        .headers(plan.auth.headers.clone())
        .build()
        .map_err(|_| Error::InvalidRequest("invalid OCR provider request".into()))?;
    send_authenticated_provider_request(
        AuthenticatedProviderRequest {
            client: &runtime.http,
            session,
            request,
            operation: if initial {
                AuthOperation::Initial
            } else {
                AuthOperation::FollowUp
            },
            deadline: attempt.deadline,
            body: if initial {
                ProviderRequestBody::Json
            } else {
                ProviderRequestBody::Empty
            },
        },
        ProviderRequest {
            api_key: plan.auth.api_key.clone(),
            provider: plan.provider.clone(),
            model: plan.model.clone(),
            body: attempt.body,
            api_base: attempt.url.to_string(),
            headers: plan
                .auth
                .headers
                .iter()
                .filter_map(|(name, value)| {
                    value
                        .to_str()
                        .ok()
                        .map(|value| (name.to_string(), value.to_string()))
                })
                .collect(),
        },
        ProviderAttemptContext {
            call_id: context.litellm_call_id.clone().unwrap_or_default(),
            trace_id: None,
            attempt: attempt.number,
        },
        observer,
    )
    .await
}

async fn poll<Observer>(
    plan: &OcrPlan,
    context: &LiteLlmRequestContext,
    runtime: &AuthRuntime,
    session: &AuthSession,
    observer: &mut Observer,
    response: ProviderHttpResponse,
    deadline: Instant,
) -> Result<Value, Error>
where
    Observer: ProviderAttemptObserver,
    Observer::Error: std::fmt::Display,
{
    let location = response
        .headers
        .get("operation-location")
        .and_then(|value| value.to_str().ok())
        .ok_or_else(|| {
            Error::InvalidResponse(
                "Azure Document Intelligence returned 202 but no Operation-Location header found"
                    .into(),
            )
        })?;
    let url = Url::parse(location)
        .map_err(|_| Error::InvalidResponse("invalid OCR operation URL".into()))?;
    session.check_destination(&url, AuthOperation::FollowUp)?;
    let mut number = 2;
    loop {
        let response = send(
            plan,
            context,
            runtime,
            session,
            observer,
            Attempt {
                url: url.clone(),
                body: BTreeMap::new(),
                number,
                deadline,
            },
        )
        .await?;
        let data: Value = serde_json::from_str(&response.body)
            .map_err(|_| Error::InvalidResponse("invalid OCR polling response JSON".into()))?;
        match data
            .get("status")
            .and_then(Value::as_str)
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("succeeded") => return Ok(data),
            Some("failed" | "canceled" | "cancelled") => {
                return Err(Error::InvalidResponse(
                    "Azure Document Intelligence operation failed".into(),
                ));
            }
            Some("notstarted" | "running") => {}
            _ => {
                return Err(Error::InvalidResponse(
                    "Azure Document Intelligence returned an invalid operation status".into(),
                ));
            }
        }
        let delay = response
            .headers
            .get("retry-after")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value >= 0.0)
            .and_then(|value| Duration::try_from_secs_f64(value).ok())
            .unwrap_or(Duration::from_secs(1));
        tokio::time::timeout_at(
            deadline.into(),
            tokio::time::sleep(delay.min(deadline.saturating_duration_since(Instant::now()))),
        )
        .await
        .map_err(|_| Error::Network("Request timed out".into()))?;
        number = number
            .checked_add(1)
            .ok_or_else(|| Error::InvalidResponse("too many OCR polling attempts".into()))?;
    }
}
