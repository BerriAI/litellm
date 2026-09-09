use std::sync::Arc;

use super::hooks::{OcrDuringCallRequest, OcrHooks, OcrLifecycleHooks};
use super::prepare::{OcrProviderRequest, PreparedOcrRequest, prepare_ocr_call};
use super::transformation::{OcrProviderConfig, request_error};
use super::types::{OcrConnection, OcrDocument, OcrRequest, OcrResponseData};
use crate::Error;
use crate::call_lifecycle::{CallLifecycle, CallLifecycleContext};
use crate::providers::azure_ai::ocr::document_intelligence::transformation::AzureDocumentIntelligenceOcrConfig;
use crate::providers::azure_ai::ocr::transformation::AzureAiOcrConfig;
use crate::providers::mistral::ocr::transformation::MistralOcrConfig;
use crate::providers::reducto::ocr::transformation::{
    ReductoParseLegacyConfig, ReductoParseV3Config,
};
use crate::providers::vertex_ai::ocr::deepseek::transformation::VertexAiDeepSeekOcrConfig;
use crate::providers::vertex_ai::ocr::transformation::VertexAiOcrConfig;

pub(crate) async fn perform_ocr_request(request: OcrRequest) -> Result<OcrResponseData, Error> {
    let context = CallLifecycleContext::new(
        "ocr",
        request.model.clone(),
        request.provider.kind().provider_name(),
        request
            .litellm_call_id
            .unwrap_or_else(|| format!("ocr-{:032x}", rand::random::<u128>())),
    );
    macro_rules! dispatch {
        ($config:expr, $params:expr) => {
            perform_provider_ocr(
                $config,
                request.model,
                request.document,
                $params,
                request.connection,
                context,
                request.hooks,
            )
            .await
        };
    }
    match request.provider {
        OcrProviderRequest::Mistral(params) => {
            dispatch!(MistralOcrConfig, params)
        }
        OcrProviderRequest::AzureAi(params) => {
            dispatch!(AzureAiOcrConfig, params)
        }
        OcrProviderRequest::AzureDocumentIntelligence(params) => {
            dispatch!(AzureDocumentIntelligenceOcrConfig, params)
        }
        OcrProviderRequest::VertexAi(params) => {
            dispatch!(VertexAiOcrConfig, params)
        }
        OcrProviderRequest::VertexAiDeepSeek(params) => {
            dispatch!(VertexAiDeepSeekOcrConfig, params)
        }
        OcrProviderRequest::ReductoV3(params) => {
            dispatch!(ReductoParseV3Config, params)
        }
        OcrProviderRequest::ReductoLegacy(params) => {
            dispatch!(ReductoParseLegacyConfig, params)
        }
    }
}

async fn perform_provider_ocr<C: OcrProviderConfig>(
    config: C,
    model: String,
    document: OcrDocument,
    params: C::InputParams,
    connection: OcrConnection,
    context: CallLifecycleContext,
    hooks: Arc<dyn OcrHooks>,
) -> Result<OcrResponseData, Error> {
    let request = prepare_ocr_call(config, model, document, params, connection);
    let lifecycle_hooks = OcrLifecycleHooks {
        hooks: hooks.clone(),
        provider_name: context.custom_llm_provider.clone(),
        marker: std::marker::PhantomData,
    };
    CallLifecycle::default()
        .run(context, request, &lifecycle_hooks, |request| {
            execute_ocr_provider_call(request, hooks.as_ref(), &lifecycle_hooks.provider_name)
        })
        .await
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
async fn execute_ocr_provider_call<C: OcrProviderConfig>(
    request: PreparedOcrRequest<C>,
    hooks: &dyn OcrHooks,
    provider_name: &str,
) -> Result<OcrResponseData, Error> {
    let config = &request.config;
    let url = config.complete_url(&request.connection, &request.model, &request.params)?;
    let headers = config.authenticate(&request.connection).await?;
    let document = if config.guard_document_before_preparation() && hooks.has_guardrails() {
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
    let document = config
        .prepare_document(document, &request.connection, &headers)
        .await?;
    let body = serde_json::to_value(config.transform_ocr_request(
        &request.model,
        document,
        &request.params,
    )?)
    .map_err(|_| request_error("body"))?;
    let body = if !config.guard_document_before_preparation() && hooks.has_guardrails() {
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
    let mut builder = super::client::http_client()?
        .post(&url)
        .json(&body)
        .timeout(request.connection.timeout);
    for (name, value) in &headers {
        builder = builder.header(name, value);
    }
    let response = crate::http_utils::http_request(builder)
        .await
        .map_err(super::client::network_error)?;
    let decoded = config
        .read_response(
            response,
            &url,
            &headers,
            &request.connection,
            &request.params,
        )
        .await?;
    let response = config.transform_ocr_response(&request.model, decoded.data, &request.params)?;
    Ok(OcrResponseData {
        provider_native_response: decoded.native,
        ..response
    })
}
