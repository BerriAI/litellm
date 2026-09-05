use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use reqwest::{RequestBuilder, StatusCode, header::HeaderMap};
use serde_json::Value;

use crate::Error;
use crate::http_utils::{http_request, truncate_error_body};
use crate::provider_callbacks::{
    CallbackDecision, ProviderAttemptObserver, ProviderError, ProviderPostCall, ProviderPreCall,
};

pub struct ProviderHttpResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: String,
}

pub struct ProviderRequest {
    pub provider: String,
    pub model: String,
    pub body: BTreeMap<String, Value>,
    pub api_base: String,
    pub headers: BTreeMap<String, String>,
}

pub struct ProviderAttemptContext {
    pub call_id: String,
    pub trace_id: Option<String>,
    pub attempt: u32,
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
pub async fn send_provider_request<Observer>(
    request: RequestBuilder,
    input: ProviderRequest,
    context: ProviderAttemptContext,
    observer: &mut Observer,
) -> Result<ProviderHttpResponse, Error>
where
    Observer: ProviderAttemptObserver,
    Observer::Error: std::fmt::Display,
{
    let event = ProviderPreCall {
        provider: input.provider,
        model: input.model,
        call_id: context.call_id,
        trace_id: context.trace_id,
        attempt: context.attempt,
        started_at: epoch_seconds(),
        request: input.body,
        api_base: input.api_base,
        headers: input.headers,
    };
    let body = match observer.pre_call(&event).await.map_err(callback_error)? {
        CallbackDecision::Unchanged => Value::Object(event.request.clone().into_iter().collect()),
        CallbackDecision::Replace { payload } => payload,
        CallbackDecision::Reject { message, .. } => return Err(Error::InvalidRequest(message)),
    };
    let response = match http_request(request.json(&body)).await {
        Ok(response) => response,
        Err(error) => {
            let mapped = transport_error(error);
            notify_error(observer, &event, &mapped, "provider_request", true).await?;
            return Err(mapped);
        }
    };
    let status = response.status();
    let headers = response.headers().clone();
    let body = match response.text().await {
        Ok(body) => body,
        Err(error) => {
            let mapped = transport_error(error);
            notify_error(observer, &event, &mapped, "response_body", true).await?;
            return Err(mapped);
        }
    };
    if !status.is_success() {
        let error = Error::Http {
            status: status.as_u16(),
            body: truncate_error_body(&body),
        };
        notify_error(observer, &event, &error, "provider_response", true).await?;
        return Err(error);
    }
    let post_call = ProviderPostCall {
        provider: event.provider.clone(),
        model: event.model.clone(),
        call_id: event.call_id.clone(),
        trace_id: event.trace_id.clone(),
        attempt: event.attempt,
        started_at: event.started_at,
        response: Value::String(body.clone()),
        status_code: status.as_u16(),
        headers: header_values(&headers),
        ended_at: epoch_seconds(),
    };
    let body = match observer
        .post_call(&post_call)
        .await
        .map_err(callback_error)?
    {
        CallbackDecision::Unchanged => body,
        CallbackDecision::Replace {
            payload: Value::String(replacement),
        } => replacement,
        CallbackDecision::Replace { payload } => {
            serde_json::to_string(&payload).map_err(|error| {
                Error::InvalidResponse(format!("callback response is invalid: {error}"))
            })?
        }
        CallbackDecision::Reject { message, .. } => return Err(Error::InvalidResponse(message)),
    };
    Ok(ProviderHttpResponse {
        status,
        headers,
        body,
    })
}

async fn notify_error<Observer>(
    observer: &mut Observer,
    context: &ProviderPreCall,
    error: &Error,
    stage: &'static str,
    committed: bool,
) -> Result<(), Error>
where
    Observer: ProviderAttemptObserver,
    Observer::Error: std::fmt::Display,
{
    let event = ProviderError {
        provider: context.provider.clone(),
        model: context.model.clone(),
        call_id: context.call_id.clone(),
        trace_id: context.trace_id.clone(),
        attempt: context.attempt,
        started_at: context.started_at,
        message: error.to_string(),
        stage,
        committed,
        status_code: match error {
            Error::Http { status, .. } => Some(*status),
            _ => None,
        },
        will_retry: false,
        ended_at: epoch_seconds(),
    };
    observer.error(&event).await.map_err(callback_error)
}

fn header_values(headers: &HeaderMap) -> BTreeMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            value
                .to_str()
                .ok()
                .map(|value| (name.to_string(), value.to_string()))
        })
        .collect()
}

fn epoch_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

fn callback_error(error: impl std::fmt::Display) -> Error {
    Error::InvalidResponse(format!("provider callback failed: {error}"))
}

fn transport_error(error: reqwest::Error) -> Error {
    Error::Network(if error.is_timeout() {
        "Request timed out".into()
    } else {
        error.to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    struct Observer {
        events: Vec<&'static str>,
        reject: bool,
    }

    impl ProviderAttemptObserver for Observer {
        type Error = std::convert::Infallible;

        async fn pre_call(
            &mut self,
            event: &ProviderPreCall,
        ) -> Result<CallbackDecision, Self::Error> {
            assert_eq!(event.provider, "test-provider");
            assert_eq!(event.model, "test-model");
            assert_eq!(event.call_id, "call-1");
            assert_eq!(event.trace_id.as_deref(), Some("trace-1"));
            assert_eq!(event.attempt, 3);
            assert_eq!(event.request["input"], "private");
            self.events.push("pre");
            Ok(if self.reject {
                CallbackDecision::Reject {
                    message: "blocked".into(),
                    status_code: Some(400),
                }
            } else {
                CallbackDecision::Replace {
                    payload: json!({"masked": true}),
                }
            })
        }

        async fn post_call(
            &mut self,
            event: &ProviderPostCall,
        ) -> Result<CallbackDecision, Self::Error> {
            assert_eq!(event.response, json!("raw-response"));
            assert_eq!(event.status_code, 200);
            assert_eq!(event.attempt, 3);
            assert_eq!(event.trace_id.as_deref(), Some("trace-1"));
            assert!(event.ended_at >= event.started_at);
            self.events.push("post");
            Ok(CallbackDecision::Replace {
                payload: json!("redacted-response"),
            })
        }

        async fn error(&mut self, event: &ProviderError) -> Result<(), Self::Error> {
            assert_eq!(event.status_code, Some(429));
            assert_eq!(event.stage, "provider_response");
            assert_eq!(event.attempt, 3);
            assert_eq!(event.trace_id.as_deref(), Some("trace-1"));
            assert!(event.committed);
            assert!(!event.will_retry);
            self.events.push("error");
            Ok(())
        }
    }

    #[tokio::test]
    async fn shared_attempt_preserves_context_decisions_and_provider_errors() {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap();
        for (status, reject) in [(200, false), (429, false), (200, true)] {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!(
                "http://{}/provider-operation",
                listener.local_addr().unwrap()
            );
            let server = tokio::spawn(async move {
                if reject {
                    assert!(
                        tokio::time::timeout(Duration::from_millis(50), listener.accept())
                            .await
                            .is_err()
                    );
                    return;
                }
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                let mut buffer = [0; 1024];
                while !request.ends_with(b"{\"masked\":true}") {
                    let count = socket.read(&mut buffer).await.unwrap();
                    assert!(count > 0);
                    request.extend_from_slice(&buffer[..count]);
                }
                let request = String::from_utf8(request).unwrap();
                assert!(request.starts_with("POST /provider-operation "));
                assert!(!request.contains("private"));
                socket.write_all(format!(
                    "HTTP/1.1 {status} Test\r\ncontent-length: 12\r\nconnection: close\r\n\r\nraw-response"
                ).as_bytes()).await.unwrap();
            });
            let mut observer = Observer {
                events: Vec::new(),
                reject,
            };
            let result = send_provider_request(
                client.post(&url),
                ProviderRequest {
                    provider: "test-provider".into(),
                    model: "test-model".into(),
                    body: BTreeMap::from([("input".into(), json!("private"))]),
                    api_base: url,
                    headers: BTreeMap::new(),
                },
                ProviderAttemptContext {
                    call_id: "call-1".into(),
                    trace_id: Some("trace-1".into()),
                    attempt: 3,
                },
                &mut observer,
            )
            .await;
            tokio::time::timeout(Duration::from_secs(2), server)
                .await
                .unwrap()
                .unwrap();
            if reject {
                assert!(
                    matches!(result, Err(Error::InvalidRequest(message)) if message == "blocked")
                );
                assert_eq!(observer.events, ["pre"]);
            } else if status == 429 {
                assert!(matches!(result, Err(Error::Http { status: 429, .. })));
                assert_eq!(observer.events, ["pre", "error"]);
            } else {
                assert_eq!(result.unwrap().body, "redacted-response");
                assert_eq!(observer.events, ["pre", "post"]);
            }
        }
    }
}
