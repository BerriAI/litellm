use super::OcrClient;
use super::adapters::OcrAdapter;
use super::hooks::OcrLifecycleHooks;
use super::registry::OcrAdapterKind;
use super::types::{LiteLLMOcrRequest, LiteLLMOcrResponse};
use crate::Error;
use crate::call_lifecycle::{CallLifecycle, CallLifecycleContext};

pub(crate) async fn perform_ocr_request(
    client: &OcrClient,
    request: LiteLLMOcrRequest,
) -> Result<LiteLLMOcrResponse, Error> {
    let context = CallLifecycleContext::new(
        "ocr",
        request.model.clone(),
        request.adapter.provider().as_str(),
        request
            .litellm_call_id
            .clone()
            .unwrap_or_else(|| format!("ocr-{:032x}", rand::random::<u128>())),
    );
    let hooks = OcrLifecycleHooks {
        hooks: request.hooks.clone(),
        provider_name: context.custom_llm_provider.clone(),
    };
    CallLifecycle::default().run(context, request, &hooks, |request| async move {
        macro_rules! execute_selected_adapter {
            ($( $variant:ident, $adapter:ty, $instance:expr, $provider:ident; )+) => {
                match request.adapter {
                    $( OcrAdapterKind::$variant => execute_ocr_provider_call(client, &$instance, request).await, )+
                }
            };
        }
        super::adapters::for_each_ocr_adapter!(execute_selected_adapter)
    }).await
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
async fn execute_ocr_provider_call<A: OcrAdapter>(
    client: &OcrClient,
    adapter: &A,
    request: LiteLLMOcrRequest,
) -> Result<LiteLLMOcrResponse, Error> {
    let provider_request = adapter.prepare_request(&request, client).await?;
    let url = provider_request.url().to_string();
    let headers = provider_request
        .headers()
        .iter()
        .map(|(name, value)| {
            value
                .to_str()
                .map(|value| (name.to_string(), value.to_string()))
                .map_err(|_| super::error::OcrRequestError::RequestField {
                    path: "headers".into(),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let response = crate::http_utils::http_request(reqwest::RequestBuilder::from_parts(
        client.provider_http().clone(),
        provider_request,
    ))
    .await
    .map_err(crate::error::TransportError::from)?;
    let decoded = adapter
        .read_response(client, response, &url, &headers, &request)
        .await?;
    let response = adapter.transform_ocr_response(&request, decoded.data)?;
    Ok(LiteLLMOcrResponse {
        provider_native_response: decoded.native,
        ..response
    })
}
