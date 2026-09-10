use std::sync::Arc;

use super::OcrClient;
use super::adapters::{MappedParams, OcrAdapter};
use super::hooks::{OcrDuringCallRequest, OcrHooks, OcrLifecycleHooks};
use super::prepare::{MappedOcrRequest, prepare_ocr_call};
use super::registry::{OcrAdapterInput, OcrAdapterRequest};
use super::types::{OcrConnection, OcrDocument, OcrRequest, OcrResponseData};
use crate::Error;
use crate::call_lifecycle::{CallLifecycle, CallLifecycleContext};

struct OcrExecution<'a> {
    client: &'a OcrClient,
    model: String,
    document: OcrDocument,
    connection: OcrConnection,
    lifecycle_context: CallLifecycleContext,
    hooks: Arc<dyn OcrHooks>,
}

pub(crate) async fn perform_ocr_request(
    client: &OcrClient,
    request: OcrRequest,
) -> Result<OcrResponseData, Error> {
    let context = CallLifecycleContext::new(
        "ocr",
        request.model.clone(),
        request.adapter.kind().provider().as_str(),
        request
            .litellm_call_id
            .unwrap_or_else(|| format!("ocr-{:032x}", rand::random::<u128>())),
    );
    let execution = OcrExecution {
        client,
        model: request.model,
        document: request.document,
        connection: request.connection,
        lifecycle_context: context,
        hooks: request.hooks,
    };
    macro_rules! execute_selected_adapter {
        ($( $variant:ident, $adapter:ty, $instance:expr, $provider:ident; )+) => {
            match request.adapter {
                $( OcrAdapterRequest::$variant(input) => execution.run($instance, input).await, )+
            }
        };
    }

    super::adapters::for_each_ocr_adapter!(execute_selected_adapter)
}

impl OcrExecution<'_> {
    async fn run<I>(self, adapter: I, input: OcrAdapterInput<I>) -> Result<OcrResponseData, Error>
    where
        I: OcrAdapter,
    {
        let Self {
            client,
            model,
            document,
            connection,
            lifecycle_context,
            hooks,
        } = self;
        let request = prepare_ocr_call(
            adapter,
            model,
            document,
            input.params,
            input.config,
            connection,
        );
        let lifecycle_hooks = OcrLifecycleHooks {
            hooks: hooks.clone(),
            provider_name: lifecycle_context.custom_llm_provider.clone(),
            marker: std::marker::PhantomData,
        };
        CallLifecycle::default()
            .run(lifecycle_context, request, &lifecycle_hooks, |request| {
                execute_ocr_provider_call(
                    client,
                    request,
                    hooks.as_ref(),
                    &lifecycle_hooks.provider_name,
                )
            })
            .await
    }
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
async fn execute_ocr_provider_call<I>(
    client: &OcrClient,
    request: MappedOcrRequest<I>,
    hooks: &dyn OcrHooks,
    provider_name: &str,
) -> Result<OcrResponseData, Error>
where
    I: OcrAdapter,
{
    let adapter = &request.adapter;
    let prepared_adapter = adapter
        .prepare(
            &request.connection,
            &request.config,
            &request.model,
            &request.params,
            &super::prepare::credential_env,
        )
        .await?;
    let url = prepared_adapter.url;
    let headers = prepared_adapter.headers;
    let body = I::build_request(&request.model, request.document, &request.params)?;
    let body = if hooks.has_guardrails() {
        let guarded = hooks
            .during_call(OcrDuringCallRequest {
                model: request.model.clone(),
                custom_llm_provider: provider_name.into(),
                url: url.clone(),
                body: serde_json::to_value(body).map_err(|_| request_error("body"))?,
            })
            .await?;
        let body: OcrWireBody<I::RequestBody> = OcrWireBody::decode(guarded.body)?;
        adapter.validate_request_body(&body.body)?;
        body
    } else {
        OcrWireBody {
            body,
            extra: serde_json::Map::new(),
        }
    };
    send_ocr_call(
        client,
        ReadyOcrCall {
            adapter: request.adapter,
            model: request.model,
            params: request.params,
            connection: request.connection,
            url,
            headers,
            body,
        },
    )
    .await
}

#[derive(serde::Serialize)]
struct OcrWireBody<B> {
    #[serde(flatten)]
    body: B,
    #[serde(flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}

impl<B: serde::Serialize + serde::de::DeserializeOwned> OcrWireBody<B> {
    fn decode(value: serde_json::Value) -> Result<Self, Error> {
        let body: B = super::wire::decode_request_value(value.clone(), "guardrail.body")?;
        let serde_json::Value::Object(fields) = value else {
            return Err(request_error("guardrail.body"));
        };
        let known = serde_json::to_value(&body).map_err(|_| request_error("guardrail.body"))?;
        let extra = fields
            .into_iter()
            .filter(|(key, _)| known.get(key).is_none())
            .collect();
        Ok(Self { body, extra })
    }
}

struct ReadyOcrCall<I: OcrAdapter> {
    adapter: I,
    model: String,
    params: MappedParams<I>,
    connection: OcrConnection,
    url: String,
    headers: Vec<(String, String)>,
    body: OcrWireBody<I::RequestBody>,
}

async fn send_ocr_call<I: OcrAdapter>(
    client: &OcrClient,
    request: ReadyOcrCall<I>,
) -> Result<OcrResponseData, Error> {
    let ReadyOcrCall {
        adapter,
        model,
        params,
        connection,
        url,
        headers,
        body,
    } = request;
    let builder = client
        .provider_http()
        .post(&url)
        .json(&body)
        .timeout(connection.timeout);
    let builder =
        crate::http_utils::with_headers(builder, &headers, crate::http_utils::HeaderPolicy::All);
    let response = crate::http_utils::http_request(builder)
        .await
        .map_err(crate::error::TransportError::from)?;
    let decoded = adapter
        .read_response(client, response, &url, &headers, &connection, &params)
        .await?;
    let response = I::normalize_response(&model, decoded.data, &params)?;
    Ok(OcrResponseData {
        provider_native_response: decoded.native,
        ..response
    })
}

fn request_error(path: &str) -> Error {
    super::error::OcrRequestError::RequestField { path: path.into() }.into()
}
