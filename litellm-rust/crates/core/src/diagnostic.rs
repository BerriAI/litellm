use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use futures_util::{Stream, stream::BoxStream};
use litellm_host::call::CallOutput;
use litellm_tracing::Logger;
use tracing::Span;

struct Completion {
    span: Span,
    outcome: &'static str,
}

impl Completion {
    fn new(name: &str) -> Self {
        let current = Span::current();
        Self {
            span: if current
                .metadata()
                .is_some_and(|metadata| metadata.name() == name)
            {
                current
            } else {
                Span::none()
            },
            outcome: "cancelled",
        }
    }

    fn finish(mut self, outcome: &'static str) {
        self.outcome = outcome;
    }
}

impl Drop for Completion {
    fn drop(&mut self) {
        self.span.record("outcome", self.outcome);
    }
}

pub(crate) fn provider(model: &str, provider: &str) {
    let span = Span::current();
    span.record("resolved_model", model);
    span.record("provider", provider);
}

pub(crate) async fn unary<R, E>(execute: impl Future<Output = Result<R, E>>) -> Result<R, E> {
    operation("litellm.route", execute).await
}

pub(crate) async fn operation<R, E>(
    name: &str,
    execute: impl Future<Output = Result<R, E>>,
) -> Result<R, E> {
    let completion = Completion::new(name);
    let result = execute.await;
    completion.finish(if result.is_ok() { "success" } else { "failure" });
    result
}

pub(crate) async fn call<R, H, C, E>(
    execute: impl Future<Output = Result<CallOutput<R, H, C, E>, E>>,
) -> Result<CallOutput<R, H, C, E>, E>
where
    C: Send + 'static,
    E: Send + 'static,
{
    let completion = Completion::new("litellm.route");
    match execute.await {
        Err(error) => {
            completion.finish("failure");
            Err(error)
        }
        Ok(CallOutput::Complete(response)) => {
            completion.span.record("stream", false);
            completion.finish("success");
            Ok(CallOutput::Complete(response))
        }
        Ok(CallOutput::Stream { head, chunks }) => {
            completion.span.record("stream", true);
            Ok(CallOutput::Stream {
                head,
                chunks: Box::pin(TracedStream {
                    state: Some(StreamState {
                        chunks,
                        completion,
                        logger: Logger::current(),
                    }),
                }),
            })
        }
    }
}

struct StreamState<C, E> {
    chunks: BoxStream<'static, Result<C, E>>,
    completion: Completion,
    logger: Logger,
}

impl<C, E> StreamState<C, E> {
    fn close(self, outcome: &'static str) {
        self.logger
            .scope(|| self.completion.span.in_scope(|| drop(self.chunks)));
        self.completion.finish(outcome);
    }
}

struct TracedStream<C, E> {
    state: Option<StreamState<C, E>>,
}

impl<C, E> Stream for TracedStream<C, E> {
    type Item = Result<C, E>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let Some(state) = self.state.as_mut() else {
            return Poll::Ready(None);
        };
        let next = state.logger.scope(|| {
            state
                .completion
                .span
                .in_scope(|| state.chunks.as_mut().poll_next(context))
        });
        let outcome = match &next {
            Poll::Ready(None) => "success",
            Poll::Ready(Some(Err(_))) => "failure",
            _ => return next,
        };
        if let Some(state) = self.state.take() {
            state.close(outcome);
        }
        next
    }
}

impl<C, E> Drop for TracedStream<C, E> {
    fn drop(&mut self) {
        if let Some(state) = self.state.take() {
            state.close("cancelled");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::mpsc, task::Context};

    use futures_util::{StreamExt, task::noop_waker_ref};
    use litellm_tracing::{Metadata, Record, Sink};
    use rstest::{fixture, rstest};
    use serde_json::{Value, json};

    use super::*;

    struct Capture(mpsc::Sender<Value>);

    impl Sink for Capture {
        fn enabled(&self, metadata: &Metadata<'_>) -> bool {
            *metadata.level() <= tracing::Level::INFO
        }
        fn emit(&self, record: &Record) {
            self.0.send(Value::Object(record.fields.clone())).unwrap();
        }
    }

    #[fixture]
    fn logger() -> (Logger, mpsc::Receiver<Value>) {
        let (sender, receiver) = mpsc::channel();
        (Logger::new(Capture(sender)), receiver)
    }

    struct Chunks(std::vec::IntoIter<Result<u8, &'static str>>);

    impl Stream for Chunks {
        type Item = Result<u8, &'static str>;
        fn poll_next(mut self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            tracing::info!(event = "poll");
            Poll::Ready(self.0.next())
        }
    }

    impl Drop for Chunks {
        fn drop(&mut self) {
            tracing::info!(event = "drop");
        }
    }

    #[tracing::instrument(
        name = "litellm.route",
        skip_all,
        fields(route = "fixture", stream, outcome)
    )]
    async fn streamed() -> Result<CallOutput<(), (), u8, &'static str>, &'static str> {
        call(async {
            Ok(CallOutput::Stream {
                head: (),
                chunks: Box::pin(Chunks(vec![Ok(1), Err("broken"), Ok(2)].into_iter())),
            })
        })
        .await
    }

    #[rstest]
    #[tokio::test]
    async fn stream_errors_finish_once_and_poll_and_drop_use_the_captured_context(
        logger: (Logger, mpsc::Receiver<Value>),
    ) {
        let (logger, records) = logger;
        let CallOutput::Stream { mut chunks, .. } = logger.instrument(streamed()).await.unwrap()
        else {
            panic!()
        };
        assert!(records.try_recv().is_err());
        tokio::spawn(async move {
            assert_eq!(chunks.next().await, Some(Ok(1)));
            assert_eq!(chunks.next().await, Some(Err("broken")));
            assert_eq!(chunks.next().await, None);
            let emitted = records.try_iter().collect::<Vec<_>>();
            assert_eq!(emitted.len(), 4);
            assert!(emitted.iter().all(|record| record["route"] == "fixture"));
            assert_eq!(emitted[0]["event"], "poll");
            assert_eq!(emitted[1]["event"], "poll");
            assert_eq!(emitted[2]["event"], "drop");
            assert_eq!(emitted[3]["outcome"], "failure");
            drop(chunks);
            assert!(records.try_recv().is_err());
        })
        .await
        .unwrap();
    }

    #[tracing::instrument(name = "litellm.route", skip_all, fields(route = "waiting", outcome))]
    async fn waiting(streaming: bool) {
        if streaming {
            let _: Result<CallOutput<(), (), u8, ()>, ()> = call(std::future::pending()).await;
        } else {
            let _: Result<(), ()> = unary(std::future::pending()).await;
        }
    }

    #[rstest]
    #[case::unary(false)]
    #[case::streaming(true)]
    fn cancellation_before_headers_closes_the_span(
        logger: (Logger, mpsc::Receiver<Value>),
        #[case] streaming: bool,
    ) {
        let (logger, records) = logger;
        let mut future = Box::pin(logger.instrument(waiting(streaming)));
        assert!(
            future
                .as_mut()
                .poll(&mut Context::from_waker(noop_waker_ref()))
                .is_pending()
        );
        assert!(records.try_recv().is_err());
        drop(future);
        let summary = records.try_recv().unwrap();
        assert_eq!(summary["outcome"], "cancelled");
        assert_eq!(summary["route"], "waiting");
        assert!(records.try_recv().is_err());
    }

    #[rstest]
    #[tokio::test]
    async fn dropped_stream_teardown_uses_its_original_logger(
        logger: (Logger, mpsc::Receiver<Value>),
    ) {
        let (logger, records) = logger;
        let output = logger.instrument(streamed()).await.unwrap();
        Logger::default().scope(|| drop(output));
        let emitted = records.try_iter().collect::<Vec<_>>();
        assert_eq!(emitted.len(), 2);
        assert_eq!(
            emitted[0],
            json!({"route":"fixture", "stream":true, "event":"drop"})
        );
        assert_eq!(emitted[1]["outcome"], "cancelled");
    }

    #[tracing::instrument(name = "disabled", level = "debug", skip_all, fields(outcome))]
    async fn disabled_child() {
        let _: Result<(), ()> = operation("disabled", async { Err(()) }).await;
    }

    #[rstest]
    #[tokio::test]
    async fn a_filtered_operation_does_not_overwrite_its_parent_outcome(
        logger: (Logger, mpsc::Receiver<Value>),
    ) {
        let (logger, records) = logger;
        logger
            .instrument(async {
                let parent = tracing::info_span!("parent", outcome = "original");
                tracing::Instrument::instrument(disabled_child(), parent).await;
            })
            .await;
        let summary = records.try_recv().unwrap();
        assert_eq!(summary["span_name"], "parent");
        assert_eq!(summary["outcome"], "original");
        assert!(records.try_recv().is_err());
    }

    #[tracing::instrument(name = "litellm.route", skip_all, fields(stream = true, outcome))]
    async fn completed() -> Result<CallOutput<(), (), u8, ()>, ()> {
        call(async { Ok(CallOutput::Complete(())) }).await
    }

    #[rstest]
    #[tokio::test]
    async fn streaming_mode_reflects_the_returned_output(logger: (Logger, mpsc::Receiver<Value>)) {
        let (logger, records) = logger;
        logger.instrument(completed()).await.unwrap();
        let summary = records.try_recv().unwrap();
        assert_eq!(summary["stream"], false);
        assert_eq!(summary["outcome"], "success");
        assert!(records.try_recv().is_err());
    }
}
