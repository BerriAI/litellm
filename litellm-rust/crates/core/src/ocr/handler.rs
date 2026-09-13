use super::OcrClient;
use super::adapters::OcrAdapter;
use super::hooks::{OcrHooks, OcrLifecycleHooks, OcrPostCallRequest};
use super::registry::OcrAdapterKind;
use super::types::{LiteLLMOcrRequest, LiteLLMOcrResponse};
use crate::Error;
use crate::call_lifecycle::{CallLifecycle, CallLifecycleContext};
use std::sync::Arc;

pub(crate) async fn perform_ocr_request(
    client: &OcrClient,
    request: LiteLLMOcrRequest,
) -> Result<LiteLLMOcrResponse, Error> {
    request.response_format()?;
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
    CallLifecycle::default()
        .run(context, request, &hooks, |request| async move {
            PreparedOcrCall::prepare(client.clone(), request)
                .await?
                .execute()
                .await?
                .normalize()
        })
        .await
}

pub(crate) struct PreparedOcrCall {
    client: OcrClient,
    request: LiteLLMOcrRequest,
    http: reqwest::Request,
}

impl PreparedOcrCall {
    pub(crate) async fn prepare(
        client: OcrClient,
        request: LiteLLMOcrRequest,
    ) -> Result<Self, Error> {
        macro_rules! prepare_adapter {
            ($( $variant:ident, $adapter:ty, $instance:expr, $provider:ident; )+) => {
                match request.adapter {
                    $( OcrAdapterKind::$variant => $instance.prepare_request(&request, &client).await?, )+
                }
            };
        }
        let http = super::adapters::for_each_ocr_adapter!(prepare_adapter);
        Ok(Self {
            client,
            request,
            http,
        })
    }

    pub(crate) async fn execute(self) -> Result<OcrProviderResponse, Error> {
        let url = self.http.url().to_string();
        let headers = request_headers(&self.http)?;
        let response = crate::http_utils::http_request(reqwest::RequestBuilder::from_parts(
            self.client.provider_http().clone(),
            self.http,
        ))
        .await
        .map_err(super::client::transport_error)?;
        macro_rules! read_adapter {
            ($( $variant:ident, $adapter:ty, $instance:expr, $provider:ident; )+) => {
                match self.request.adapter {
                    $( OcrAdapterKind::$variant => {
                        let decoded = $instance.read_response(&self.client, response, &url, &headers, &self.request).await?;
                        Ok(OcrProviderResponse {
                            request: self.request,
                            data: OcrProviderData::$variant(decoded),
                        })
                    }, )+
                }
            };
        }
        super::adapters::for_each_ocr_adapter!(read_adapter)
    }
}

fn request_headers(request: &reqwest::Request) -> Result<Vec<(String, String)>, Error> {
    request
        .headers()
        .iter()
        .map(|(name, value)| {
            value
                .to_str()
                .map(|value| (name.to_string(), value.to_string()))
                .map_err(|_| super::error::OcrRequestError::RequestField {
                    path: "headers".into(),
                })
                .map_err(Error::from)
        })
        .collect()
}

macro_rules! provider_data {
    ($( $variant:ident, $adapter:ty, $instance:expr, $provider:ident; )+) => {
        enum OcrProviderData {
            $( $variant(super::wire::DecodedOcrResponse<<$adapter as OcrAdapter>::ProviderResponse>), )+
        }

        impl OcrProviderResponse {
            pub(crate) fn normalize(self) -> Result<LiteLLMOcrResponse, Error> {
                match self.data {
                    $( OcrProviderData::$variant(decoded) => {
                        let response = $instance.transform_ocr_response(&self.request, decoded.data)?;
                        Ok(LiteLLMOcrResponse { provider_native_response: decoded.native, ..response })
                    }, )+
                }
            }
        }
    };
}

pub(crate) struct OcrProviderResponse {
    request: LiteLLMOcrRequest,
    data: OcrProviderData,
}

pub(crate) async fn post_call(hooks: &Arc<dyn OcrHooks>, bytes: &[u8]) -> Result<(), Error> {
    let original_response = serde_json::Value::String(String::from_utf8_lossy(bytes).into_owned());
    hooks
        .post_call(OcrPostCallRequest { original_response })
        .await?;
    Ok(())
}

super::adapters::for_each_ocr_adapter!(provider_data);
