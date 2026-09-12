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

pub struct PreparedOcrCall {
    client: OcrClient,
    request: LiteLLMOcrRequest,
    pub http: reqwest::Request,
}

impl PreparedOcrCall {
    pub fn provider(&self) -> &str {
        self.request.adapter.provider().as_str()
    }

    pub fn request(&self) -> &LiteLLMOcrRequest {
        &self.request
    }

    pub async fn new(request: LiteLLMOcrRequest) -> Result<Self, Error> {
        Self::prepare(super::client::shared_client()?, request).await
    }

    async fn prepare(client: OcrClient, request: LiteLLMOcrRequest) -> Result<Self, Error> {
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

    pub async fn execute(self) -> Result<OcrProviderResponse, Error> {
        let url = self.http.url().to_string();
        let headers = self
            .http
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
            self.client.provider_http().clone(),
            self.http,
        ))
        .await
        .map_err(super::client::transport_error)?;
        macro_rules! read_adapter {
            ($( $variant:ident, $adapter:ty, $instance:expr, $provider:ident; )+) => {
                match self.request.adapter {
                    $( OcrAdapterKind::$variant => {
                        let bytes = $instance.read_response(&self.client, response, &url, &headers, &self.request).await?;
                        Ok(OcrProviderResponse {
                            text: String::from_utf8_lossy(&bytes).into_owned(),
                            request: self.request,
                            bytes,
                        })
                    }, )+
                }
            };
        }
        super::adapters::for_each_ocr_adapter!(read_adapter)
    }
}

macro_rules! provider_data {
    ($( $variant:ident, $adapter:ty, $instance:expr, $provider:ident; )+) => {
        impl OcrProviderResponse {
            pub fn normalize(self) -> Result<LiteLLMOcrResponse, Error> {
                let native = self.request.response_format()? == super::types::OcrResponseFormat::Native;
                match self.request.adapter {
                    $( OcrAdapterKind::$variant => {
                        let decoded = super::wire::decode_response::<<$adapter as OcrAdapter>::ProviderResponse>(&self.bytes, native)?;
                        let response = $instance.transform_ocr_response(&self.request, decoded.data)?;
                        Ok(LiteLLMOcrResponse { provider_native_response: decoded.native, ..response })
                    }, )+
                }
            }
        }
    };
}

pub struct OcrProviderResponse {
    pub text: String,
    request: LiteLLMOcrRequest,
    bytes: Vec<u8>,
}

super::adapters::for_each_ocr_adapter!(provider_data);
