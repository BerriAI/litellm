use std::future::{Future, ready};
use std::pin::Pin;

use litellm_core::call_lifecycle::{
    CallLifecycle, CallLifecycleContext, CallLifecycleHooks, CallLifecycleTiming,
};
use litellm_core::{CoreError, CoreResult};
use serde_json::Value;

mod client;
mod common_utils;
mod handler;
mod prepare;
mod provider;
mod types;

pub use prepare::{ocr_decline_reason, prepare_ocr_request, prepare_provider_request};
pub use types::{OcrDeclineReason, OcrRequest, PreparedOcrRequest, ProviderOcrRequest};

use handler::execute_ocr_provider_call;

pub async fn ocr(request: OcrRequest<'_>) -> CoreResult<Value> {
    ocr_with_hooks(prepare_ocr_request(request), &NoopOcrLifecycleHooks).await
}

pub async fn ocr_with_hooks<Hooks>(request: PreparedOcrRequest, hooks: &Hooks) -> CoreResult<Value>
where
    Hooks: CallLifecycleHooks<PreparedOcrRequest, ProviderOcrRequest, Value>,
{
    CallLifecycle::default()
        .run_request(request, hooks, execute_ocr_provider_call)
        .await
}

pub struct NoopOcrLifecycleHooks;

impl CallLifecycleHooks<PreparedOcrRequest, ProviderOcrRequest, Value> for NoopOcrLifecycleHooks {
    type PreCallFuture<'a> = std::future::Ready<CoreResult<PreparedOcrRequest>>;
    type DuringCallFuture<'a> =
        Pin<Box<dyn Future<Output = CoreResult<ProviderOcrRequest>> + Send + 'a>>;
    type SuccessFuture<'a> = std::future::Ready<()>;
    type FailureFuture<'a> = std::future::Ready<()>;

    fn async_pre_call_hook<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        request: PreparedOcrRequest,
    ) -> Self::PreCallFuture<'a> {
        ready(Ok(request))
    }

    fn async_during_call_hook<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        request: PreparedOcrRequest,
    ) -> Self::DuringCallFuture<'a> {
        Box::pin(prepare_provider_request(request))
    }

    fn async_log_success_event<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        _response: &'a Value,
        _timing: &'a CallLifecycleTiming,
    ) -> Self::SuccessFuture<'a> {
        ready(())
    }

    fn async_log_failure_event<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        _error: &'a CoreError,
        _timing: &'a CallLifecycleTiming,
    ) -> Self::FailureFuture<'a> {
        ready(())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, json};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;

    #[tokio::test]
    async fn direct_runtime_ocr_uses_noop_lifecycle() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = vec![0_u8; 4096];
            let _ = socket.read(&mut request).await.unwrap();
            let body = json!({
                "pages": [{"index": 0, "markdown": "runtime"}],
                "model": "mistral-ocr-latest",
                "usage_info": {"pages_processed": 1}
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });

        let api_base = format!("http://{address}");
        let response = ocr(OcrRequest {
            model: "mistral/mistral-ocr-latest",
            document: json!({
                "type": "document_url",
                "document_url": "https://example.com/document.pdf"
            }),
            api_key: Some("test-key"),
            api_base: Some(&api_base),
            custom_llm_provider: None,
            extra_headers: None,
            optional_params: Map::new(),
            timeout: None,
            litellm_call_id: Some("runtime-test"),
        })
        .await
        .unwrap();

        server.await.unwrap();
        assert_eq!(response["pages"][0]["markdown"], "runtime");
    }
}
