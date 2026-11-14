use std::{any::Any, fmt, future::Future, sync::Arc};

use litellm_coroutine::{Abandoned, Co, Coroutine, Reply};
use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

pub trait HostCall: Serialize + Send + 'static {
    const NAME: &'static str;
    type Reply: DeserializeOwned + Send + 'static;
}

#[derive(Clone)]
pub struct HostException {
    message: String,
    original: Arc<dyn Any + Send + Sync>,
}

impl HostException {
    pub fn new(message: impl Into<String>, original: impl Any + Send + Sync) -> Self {
        Self {
            message: message.into(),
            original: Arc::new(original),
        }
    }

    pub fn message(&self) -> &str {
        &self.message
    }

    pub fn original<T: Any>(&self) -> Option<&T> {
        self.original.downcast_ref()
    }
}

impl fmt::Debug for HostException {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HostException")
            .field("message", &self.message)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for HostException {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

#[derive(Debug)]
pub enum HostFault {
    Raised(HostException),
    Malformed(String),
}

pub type HostReply = Result<Value, HostFault>;

#[derive(Debug)]
pub struct HostRequest {
    pub call: &'static str,
    pub request: Value,
    pub reply: Reply<HostReply>,
}

#[derive(Debug, thiserror::Error)]
pub enum HostCallError {
    #[error("no host serves the {call} host call")]
    Detached { call: &'static str },
    #[error("the {call} host call request does not serialise: {source}")]
    Request {
        call: &'static str,
        source: serde_json::Error,
    },
    #[error("the {call} host call raised: {exception}")]
    Raised {
        call: &'static str,
        exception: HostException,
    },
    #[error("the {call} host call replied with an unexpected value: {message}")]
    Reply { call: &'static str, message: String },
    #[error("the {call} host call was abandoned")]
    Abandoned { call: &'static str },
}

impl HostCallError {
    pub fn exception(&self) -> Option<&HostException> {
        match self {
            Self::Raised { exception, .. } => Some(exception),
            _ => None,
        }
    }
}

#[derive(Clone, Default)]
pub struct Host {
    co: Option<Co<HostRequest>>,
}

impl Host {
    pub fn detached() -> Self {
        Self::default()
    }

    pub fn attached(co: Co<HostRequest>) -> Self {
        Self { co: Some(co) }
    }

    pub const fn is_attached(&self) -> bool {
        self.co.is_some()
    }

    pub async fn call<C: HostCall>(&self, call: C) -> Result<C::Reply, HostCallError> {
        let name = C::NAME;
        let Some(co) = &self.co else {
            return Err(HostCallError::Detached { call: name });
        };
        let request = serde_json::to_value(&call)
            .map_err(|source| HostCallError::Request { call: name, source })?;
        let answer = co
            .yield_(|reply| HostRequest {
                call: name,
                request,
                reply,
            })
            .await
            .map_err(|Abandoned| HostCallError::Abandoned { call: name })?;
        match answer {
            Ok(value) => serde_json::from_value(value).map_err(|error| HostCallError::Reply {
                call: name,
                message: error.to_string(),
            }),
            Err(HostFault::Raised(exception)) => Err(HostCallError::Raised {
                call: name,
                exception,
            }),
            Err(HostFault::Malformed(message)) => Err(HostCallError::Reply {
                call: name,
                message,
            }),
        }
    }
}

impl fmt::Debug for Host {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Host")
            .field("attached", &self.is_attached())
            .finish()
    }
}

pub type Hosted<C> = Coroutine<HostRequest, C>;

pub fn hosted<C, F>(run: impl FnOnce(Host) -> F) -> Hosted<C>
where
    F: Future<Output = C> + Send + 'static,
{
    Coroutine::new(|co| run(Host::attached(co)))
}

#[cfg(test)]
mod tests {
    use futures_util::FutureExt;
    use litellm_coroutine::{CoroutineState, ResumeError};
    use rstest::rstest;
    use serde::Deserialize;
    use serde_json::json;

    use super::*;

    #[derive(Serialize)]
    struct CurrentSpend {
        key: String,
    }

    #[derive(Debug, PartialEq, Deserialize)]
    struct Spend {
        spend: f64,
    }

    impl HostCall for CurrentSpend {
        const NAME: &'static str = "current_spend";
        type Reply = Spend;
    }

    #[derive(Serialize)]
    struct Reseed;

    impl HostCall for Reseed {
        const NAME: &'static str = "reseed_counter";
        type Reply = ();
    }

    fn step<C>(hosted: &mut Hosted<C>) -> CoroutineState<HostRequest, C> {
        hosted.resume().now_or_never().unwrap().unwrap()
    }

    fn yielded<C>(hosted: &mut Hosted<C>) -> HostRequest {
        match step(hosted) {
            CoroutineState::Yielded(request) => request,
            CoroutineState::Complete(_) => panic!("the body completed without a host call"),
        }
    }

    fn completed<C>(hosted: &mut Hosted<C>) -> C {
        match step(hosted) {
            CoroutineState::Complete(output) => output,
            CoroutineState::Yielded(request) => panic!("unexpected host call {}", request.call),
        }
    }

    fn spend_of(key: &str) -> Hosted<Result<Spend, HostCallError>> {
        let key = key.to_owned();
        hosted(|host| async move { host.call(CurrentSpend { key }).await })
    }

    #[rstest]
    fn a_typed_call_yields_its_name_and_request_and_decodes_the_typed_reply() {
        let mut hosted = spend_of("sk-1");
        let request = yielded(&mut hosted);
        assert_eq!(request.call, "current_spend");
        assert_eq!(request.request, json!({"key": "sk-1"}));
        request.reply.send(Ok(json!({"spend": 2.5})));
        assert_eq!(completed(&mut hosted).unwrap(), Spend { spend: 2.5 });
    }

    #[rstest]
    fn a_detached_host_fails_the_call_without_yielding() {
        let error = Host::detached()
            .call(CurrentSpend { key: "sk".into() })
            .now_or_never()
            .unwrap()
            .unwrap_err();
        assert!(matches!(
            error,
            HostCallError::Detached {
                call: "current_spend"
            }
        ));
        assert_eq!(
            error.to_string(),
            "no host serves the current_spend host call"
        );
    }

    #[rstest]
    fn a_raised_host_call_carries_the_original_exception() {
        let mut hosted = spend_of("sk-1");
        yielded(&mut hosted)
            .reply
            .send(Err(HostFault::Raised(HostException::new(
                "KeyError: 'sk-1'",
                7_u32,
            ))));
        let error = completed(&mut hosted).unwrap_err();
        let exception = error.exception().unwrap();
        assert_eq!(exception.message(), "KeyError: 'sk-1'");
        assert_eq!(exception.original::<u32>(), Some(&7));
        assert_eq!(exception.original::<String>(), None);
        assert_eq!(
            error.to_string(),
            "the current_spend host call raised: KeyError: 'sk-1'"
        );
    }

    #[rstest]
    #[case::wrong_shape(Ok(json!("lots")))]
    #[case::malformed(Err(HostFault::Malformed("not json".into())))]
    fn a_reply_the_call_cannot_decode_is_a_reply_error(#[case] reply: HostReply) {
        let mut hosted = spend_of("sk-1");
        yielded(&mut hosted).reply.send(reply);
        assert!(matches!(
            completed(&mut hosted).unwrap_err(),
            HostCallError::Reply {
                call: "current_spend",
                ..
            }
        ));
    }

    #[rstest]
    fn a_dropped_reply_abandons_the_call() {
        let mut hosted = spend_of("sk-1");
        drop(yielded(&mut hosted).reply);
        assert!(matches!(
            completed(&mut hosted).unwrap_err(),
            HostCallError::Abandoned {
                call: "current_spend"
            }
        ));
    }

    #[rstest]
    fn sequential_calls_are_answered_in_order_and_the_body_sees_each_reply() {
        let mut hosted = hosted(|host| async move {
            let first = host.call(CurrentSpend { key: "a".into() }).await?;
            host.call(Reseed).await?;
            let second = host.call(CurrentSpend { key: "b".into() }).await?;
            Ok::<_, HostCallError>(first.spend + second.spend)
        });
        let first = yielded(&mut hosted);
        assert_eq!(first.request, json!({"key": "a"}));
        first.reply.send(Ok(json!({"spend": 1.0})));
        let reseed = yielded(&mut hosted);
        assert_eq!(
            (reseed.call, reseed.request.clone()),
            ("reseed_counter", Value::Null)
        );
        reseed.reply.send(Ok(Value::Null));
        let second = yielded(&mut hosted);
        assert_eq!(second.request, json!({"key": "b"}));
        second.reply.send(Ok(json!({"spend": 2.0})));
        assert_eq!(completed(&mut hosted).unwrap(), 3.0);
    }

    #[rstest]
    fn concurrent_calls_surface_one_at_a_time() {
        let mut hosted = hosted(|host| async move {
            let (first, second) = futures_util::future::join(
                host.call(CurrentSpend { key: "a".into() }),
                host.call(CurrentSpend { key: "b".into() }),
            )
            .await;
            (first.unwrap().spend, second.unwrap().spend)
        });
        let first = yielded(&mut hosted);
        assert!(matches!(
            hosted.resume().now_or_never().unwrap(),
            Err(ResumeError::Unanswered)
        ));
        first.reply.send(Ok(json!({"spend": 1.0})));
        let second = yielded(&mut hosted);
        assert_eq!(second.request, json!({"key": "b"}));
        second.reply.send(Ok(json!({"spend": 2.0})));
        assert_eq!(completed(&mut hosted), (1.0, 2.0));
    }

    #[rstest]
    fn cancelling_the_body_abandons_the_outstanding_call_and_drops_the_body() {
        struct Dropped(Arc<std::sync::atomic::AtomicBool>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let guard = Dropped(Arc::clone(&dropped));
        let mut hosted = hosted(|host| async move {
            let _guard = guard;
            host.call(Reseed).await
        });
        let request = yielded(&mut hosted);
        hosted.cancel();
        assert!(dropped.load(std::sync::atomic::Ordering::SeqCst));
        request.reply.send(Ok(Value::Null));
    }
}
