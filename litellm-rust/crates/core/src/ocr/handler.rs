use std::sync::Arc;

use super::hooks::{OcrDuringCallRequest, OcrHooks, OcrLifecycleHooks};
use super::prepare::{OcrProviderRequest, PreparedOcrRequest, prepare_ocr_call};
use super::transformation::{OcrBackend, OcrFormat, request_error};
use super::types::{OcrConnection, OcrDocument, OcrRequest, OcrResponseData};
use crate::Error;
use crate::call_lifecycle::{CallLifecycle, CallLifecycleContext};
use crate::providers::azure_ai::mistral::ocr::transformation::AzureMistralOcrBackend;
use crate::providers::azure_ai::ocr::document_intelligence::transformation::AzureDocumentIntelligenceOcrBackend;
use crate::providers::mistral::ocr::transformation::MistralOcrBackend;
use crate::providers::reducto::ocr::transformation::{
    ReductoParseLegacyBackend, ReductoParseV3Backend,
};
use crate::providers::vertex_ai::deepseek::ocr::transformation::VertexDeepSeekOcrBackend;
use crate::providers::vertex_ai::mistral::ocr::transformation::VertexMistralOcrBackend;

struct OcrExecution<'a> {
    http_client: &'a reqwest::Client,
    lifecycle_context: CallLifecycleContext,
    hooks: Arc<dyn OcrHooks>,
}

pub(crate) async fn perform_ocr_request(
    http_client: &reqwest::Client,
    request: OcrRequest,
) -> Result<OcrResponseData, Error> {
    let context = CallLifecycleContext::new(
        "ocr",
        request.model.clone(),
        request.provider.kind().provider_name(),
        request
            .litellm_call_id
            .unwrap_or_else(|| format!("ocr-{:032x}", rand::random::<u128>())),
    );
    macro_rules! dispatch {
        ($backend:expr, $params:expr) => {
            perform_provider_ocr(
                $backend,
                request.model,
                request.document,
                $params,
                request.connection,
                OcrExecution {
                    http_client,
                    lifecycle_context: context,
                    hooks: request.hooks,
                },
            )
            .await
        };
    }
    match request.provider {
        OcrProviderRequest::Mistral(params) => {
            dispatch!(MistralOcrBackend, params)
        }
        OcrProviderRequest::AzureAi(params) => {
            dispatch!(AzureMistralOcrBackend, params)
        }
        OcrProviderRequest::AzureDocumentIntelligence(params) => {
            dispatch!(AzureDocumentIntelligenceOcrBackend, params)
        }
        OcrProviderRequest::VertexAi(params) => {
            dispatch!(VertexMistralOcrBackend, params)
        }
        OcrProviderRequest::VertexAiDeepSeek(params) => {
            dispatch!(VertexDeepSeekOcrBackend, params)
        }
        OcrProviderRequest::ReductoV3(params) => {
            dispatch!(ReductoParseV3Backend, params)
        }
        OcrProviderRequest::ReductoLegacy(params) => {
            dispatch!(ReductoParseLegacyBackend, params)
        }
    }
}

async fn perform_provider_ocr<C: OcrBackend>(
    backend: C,
    model: String,
    document: OcrDocument,
    params: <C::Format as OcrFormat>::InputParams,
    connection: OcrConnection,
    execution: OcrExecution<'_>,
) -> Result<OcrResponseData, Error> {
    let OcrExecution {
        http_client,
        lifecycle_context,
        hooks,
    } = execution;
    let request = prepare_ocr_call(backend, model, document, params, connection);
    let lifecycle_hooks = OcrLifecycleHooks {
        hooks: hooks.clone(),
        provider_name: lifecycle_context.custom_llm_provider.clone(),
        marker: std::marker::PhantomData,
    };
    CallLifecycle::default()
        .run(lifecycle_context, request, &lifecycle_hooks, |request| {
            execute_ocr_provider_call(
                http_client,
                request,
                hooks.as_ref(),
                &lifecycle_hooks.provider_name,
            )
        })
        .await
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
async fn execute_ocr_provider_call<C: OcrBackend>(
    http_client: &reqwest::Client,
    request: PreparedOcrRequest<C>,
    hooks: &dyn OcrHooks,
    provider_name: &str,
) -> Result<OcrResponseData, Error> {
    let backend = &request.backend;
    let url = backend.complete_url(&request.connection, &request.model, &request.params)?;
    let headers = backend.authenticate(&request.connection).await?;
    let document = if backend.guard_document_before_preparation() && hooks.has_guardrails() {
        let guarded = hooks
            .during_call(OcrDuringCallRequest {
                model: request.model.clone(),
                custom_llm_provider: provider_name.into(),
                url: url.clone(),
                body: serde_json::to_value(request.document)
                    .map_err(|_| request_error("document"))?,
            })
            .await?;
        super::wire::decode_request_value(guarded.body, "guardrail.body")?
    } else {
        request.document
    };
    let document = backend
        .prepare_document(http_client, document, &request.connection, &headers)
        .await?;
    let body = serde_json::to_value(backend.format().transform_ocr_request(
        &request.model,
        document,
        &request.params,
    )?)
    .map_err(|_| request_error("body"))?;
    let body = if !backend.guard_document_before_preparation() && hooks.has_guardrails() {
        let guarded = hooks
            .during_call(OcrDuringCallRequest {
                model: request.model.clone(),
                custom_llm_provider: provider_name.into(),
                url: url.clone(),
                body,
            })
            .await?;
        guarded.body
    } else {
        body
    };
    let mut builder = http_client
        .post(&url)
        .json(&body)
        .timeout(request.connection.timeout);
    for (name, value) in &headers {
        builder = builder.header(name, value);
    }
    let response = crate::http_utils::http_request(builder)
        .await
        .map_err(super::client::network_error)?;
    let decoded = backend
        .read_response(
            http_client,
            response,
            &url,
            &headers,
            &request.connection,
            &request.params,
        )
        .await?;
    let response =
        backend
            .format()
            .transform_ocr_response(&request.model, decoded.data, &request.params)?;
    Ok(OcrResponseData {
        provider_native_response: decoded.native,
        ..response
    })
}
