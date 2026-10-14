use super::{CallHooks, DynCallHooks, HookRuntime, RuntimeCallEvent};
use crate::{
    interceptors::{RequestContext, WireRequest},
    lifecycle::{CallEvent, Timing},
};

pub trait ChainRuntime: HookRuntime + Sized + 'static {
    type Group: DynCallHooks<Self> + ?Sized;

    /// `Ok` lets the next group receive the terminal event, `Err` stops the chain.
    fn terminal_notification_failed(error: Self::Error) -> Result<(), Self::Error>;
}

pub trait HookGroupAdapter<H>: ChainRuntime {
    fn adapt_group(hooks: H) -> Box<Self::Group>;
}

pub struct HookChain<R: ChainRuntime> {
    hooks: Vec<Box<R::Group>>,
}

impl<R: ChainRuntime> HookChain<R> {
    pub fn groups(&self) -> impl Iterator<Item = &R::Group> {
        self.hooks.iter().map(Box::as_ref)
    }

    pub fn groups_mut(&mut self) -> impl Iterator<Item = &mut R::Group> {
        self.hooks.iter_mut().map(Box::as_mut)
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn with<H>(mut self, hooks: H) -> Self
    where
        R: HookGroupAdapter<H>,
    {
        self.hooks.push(R::adapt_group(hooks));
        self
    }
}

impl<R: ChainRuntime> CallHooks<R> for HookChain<R>
where
    for<'a> R::Context<'a>: Copy,
{
    async fn prepare_arguments<'a>(
        &'a self,
        runtime: R::Context<'a>,
        arguments: R::Arguments,
        started_at: f64,
    ) -> Result<R::Arguments, R::Error> {
        let mut arguments = arguments;
        for hooks in &self.hooks {
            arguments = hooks
                .prepare_arguments(runtime, arguments, started_at)
                .await?;
        }
        Ok(arguments)
    }

    fn arguments_prepared(&self, arguments: &R::Arguments) -> Result<(), R::Error> {
        self.hooks
            .iter()
            .try_for_each(|hooks| hooks.arguments_prepared(arguments))
    }

    async fn before_provider_request<'a>(
        &'a self,
        runtime: R::Context<'a>,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> Result<Box<WireRequest>, R::Error> {
        let mut wire = wire;
        for hooks in &self.hooks {
            wire = hooks
                .before_provider_request(runtime, wire, context)
                .await?;
        }
        Ok(wire)
    }

    async fn transform_response<'a>(
        &'a self,
        runtime: R::Context<'a>,
        response: R::Response,
        timing: Timing,
    ) -> Result<R::Response, R::Error> {
        let mut response = response;
        for hooks in &self.hooks {
            response = hooks.transform_response(runtime, response, timing).await?;
        }
        Ok(response)
    }

    async fn on_event<'a>(
        &'a self,
        runtime: R::Context<'a>,
        event: RuntimeCallEvent<'_, R>,
    ) -> Result<(), R::Error> {
        let terminal = matches!(
            event,
            CallEvent::Succeeded { .. } | CallEvent::Failed { .. }
        );
        for hooks in &self.hooks {
            match hooks.on_event(runtime, event.clone()).await {
                Err(error) if terminal => R::terminal_notification_failed(error)?,
                result => result?,
            }
        }
        Ok(())
    }

    fn on_stream_open(&self, head: &R::Response) -> Result<(), R::Error> {
        self.hooks
            .iter()
            .try_for_each(|hooks| hooks.on_stream_open(head))
    }

    fn on_stream_chunk(&self, chunk: &R::Chunk) -> Result<(), R::Error> {
        self.hooks
            .iter()
            .try_for_each(|hooks| hooks.on_stream_chunk(chunk))
    }
}

impl<R: ChainRuntime> Default for HookChain<R> {
    fn default() -> Self {
        Self { hooks: Vec::new() }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use rstest::{fixture, rstest};

    use super::*;
    use crate::lifecycle::FailureOrigin;

    struct TestRuntime;

    impl HookRuntime for TestRuntime {
        type Context<'a> = &'a Mutex<Vec<String>>;
        type Arguments = String;
        type Response = String;
        type Chunk = u32;
        type Error = &'static str;
        type Failure = &'static str;
    }

    impl ChainRuntime for TestRuntime {
        type Group = dyn DynCallHooks<Self>;

        fn terminal_notification_failed(error: &'static str) -> Result<(), &'static str> {
            match error {
                "cancelled" => Err(error),
                _ => Ok(()),
            }
        }
    }

    impl<H: CallHooks<TestRuntime> + 'static> HookGroupAdapter<H> for TestRuntime {
        fn adapt_group(hooks: H) -> Box<Self::Group> {
            Box::new(hooks)
        }
    }

    struct Group {
        name: &'static str,
        suspensions: u8,
        reject: Option<&'static str>,
    }

    impl Group {
        async fn value(
            &self,
            trace: &Mutex<Vec<String>>,
            stage: &str,
            value: String,
        ) -> Result<String, &'static str> {
            trace.lock().unwrap().push(format!("{stage}:{}", self.name));
            for _ in 0..self.suspensions {
                tokio::task::yield_now().await;
            }
            match self.reject {
                Some(error) => Err(error),
                None => Ok(format!("{value}{}", self.name)),
            }
        }
    }

    impl CallHooks<TestRuntime> for Group {
        async fn prepare_arguments<'a>(
            &'a self,
            trace: &'a Mutex<Vec<String>>,
            arguments: String,
            _: f64,
        ) -> Result<String, &'static str> {
            self.value(trace, "prepare", arguments).await
        }

        async fn before_provider_request<'a>(
            &'a self,
            trace: &'a Mutex<Vec<String>>,
            wire: Box<WireRequest>,
            _: &RequestContext,
        ) -> Result<Box<WireRequest>, &'static str> {
            let url = self.value(trace, "wire", wire.url.clone()).await?;
            Ok(Box::new(WireRequest { url, ..*wire }))
        }

        async fn transform_response<'a>(
            &'a self,
            trace: &'a Mutex<Vec<String>>,
            response: String,
            _: Timing,
        ) -> Result<String, &'static str> {
            self.value(trace, "response", response).await
        }

        async fn on_event<'a>(
            &'a self,
            trace: &'a Mutex<Vec<String>>,
            event: RuntimeCallEvent<'_, TestRuntime>,
        ) -> Result<(), &'static str> {
            let label = match event {
                CallEvent::Succeeded { response, .. } => response.as_str(),
                CallEvent::Failed { error, .. } => *error,
                _ => "event",
            };
            self.value(trace, label, String::new()).await.map(|_| ())
        }
    }

    #[fixture]
    fn trace() -> Mutex<Vec<String>> {
        Mutex::new(Vec::new())
    }

    #[fixture]
    fn timing() -> Timing {
        Timing {
            start_time: 0.0,
            end_time: 1.0,
        }
    }

    #[fixture]
    fn context() -> RequestContext {
        RequestContext {
            model: String::new(),
            custom_llm_provider: String::new(),
            optional_params: serde_json::Value::Null,
            secret_fields: Vec::new(),
            api_key: None,
        }
    }

    #[rstest]
    #[tokio::test]
    async fn groups_compose_without_python(
        trace: Mutex<Vec<String>>,
        timing: Timing,
        context: RequestContext,
        #[values(0, 1, 2)] suspensions: u8,
    ) {
        let hooks = HookChain::<TestRuntime>::new()
            .with(Group {
                name: "a",
                suspensions,
                reject: None,
            })
            .with(Group {
                name: "b",
                suspensions,
                reject: None,
            });
        let arguments = CallHooks::prepare_arguments(&hooks, &trace, "input".into(), 0.0)
            .await
            .unwrap();
        assert_eq!(arguments, "inputab");
        let wire = Box::new(WireRequest {
            url: "url".into(),
            headers: Vec::new(),
            body: serde_json::Value::Null,
        });
        let rewritten = CallHooks::before_provider_request(&hooks, &trace, wire, &context)
            .await
            .unwrap();
        assert_eq!(rewritten.url, "urlab");
        let response = CallHooks::transform_response(&hooks, &trace, "output".into(), timing)
            .await
            .unwrap();
        assert_eq!(response, "outputab");
        CallHooks::on_event(
            &hooks,
            &trace,
            CallEvent::Succeeded {
                timing,
                response: &response,
            },
        )
        .await
        .unwrap();
        assert_eq!(
            *trace.lock().unwrap(),
            [
                "prepare:a",
                "prepare:b",
                "wire:a",
                "wire:b",
                "response:a",
                "response:b",
                "outputab:a",
                "outputab:b",
            ]
        );
    }

    fn rejecting_chain(suspensions: u8, error: &'static str) -> HookChain<TestRuntime> {
        ["a", "b", "c"]
            .into_iter()
            .fold(HookChain::new(), |hooks, name| {
                hooks.with(Group {
                    name,
                    suspensions,
                    reject: (name == "b").then_some(error),
                })
            })
    }

    #[rstest]
    #[tokio::test]
    async fn transformation_failure_stops_later_groups(
        trace: Mutex<Vec<String>>,
        #[values(0, 2)] suspensions: u8,
    ) {
        let hooks = rejecting_chain(suspensions, "rejected");
        let result = CallHooks::prepare_arguments(&hooks, &trace, String::new(), 0.0).await;
        assert_eq!(result, Err("rejected"));
        assert_eq!(*trace.lock().unwrap(), ["prepare:a", "prepare:b"]);
    }

    #[rstest]
    #[case::success_reaches_every_group(true, "rejected", Ok(()), &["selected:a", "selected:b", "selected:c"])]
    #[case::failure_reaches_every_group(false, "rejected", Ok(()), &["failed:a", "failed:b", "failed:c"])]
    #[case::cancellation_stops_success(true, "cancelled", Err("cancelled"), &["selected:a", "selected:b"])]
    #[case::cancellation_stops_failure(false, "cancelled", Err("cancelled"), &["failed:a", "failed:b"])]
    #[tokio::test]
    async fn terminal_notification_errors_follow_the_runtime(
        trace: Mutex<Vec<String>>,
        timing: Timing,
        #[values(0, 2)] suspensions: u8,
        #[case] succeeded: bool,
        #[case] error: &'static str,
        #[case] expected: Result<(), &'static str>,
        #[case] reached: &[&str],
    ) {
        let hooks = rejecting_chain(suspensions, error);
        let response = "selected".to_owned();
        let event = if succeeded {
            CallEvent::Succeeded {
                timing,
                response: &response,
            }
        } else {
            CallEvent::Failed {
                timing,
                origin: FailureOrigin::Host,
                error: &"failed",
            }
        };
        let result = CallHooks::on_event(&hooks, &trace, event).await;
        assert_eq!(result, expected);
        assert_eq!(*trace.lock().unwrap(), reached);
    }

    #[rstest]
    #[tokio::test]
    async fn non_terminal_notification_error_stops_later_groups(trace: Mutex<Vec<String>>) {
        let hooks = rejecting_chain(0, "rejected");
        let result =
            CallHooks::on_event(&hooks, &trace, CallEvent::Started { start_time: 0.0 }).await;
        assert_eq!(result, Err("rejected"));
        assert_eq!(*trace.lock().unwrap(), ["event:a", "event:b"]);
    }

    #[rstest]
    #[tokio::test]
    async fn empty_chain_preserves_values(trace: Mutex<Vec<String>>, timing: Timing) {
        let hooks = HookChain::<TestRuntime>::new();
        assert_eq!(
            CallHooks::prepare_arguments(&hooks, &trace, "input".into(), 0.0)
                .await
                .unwrap(),
            "input"
        );
        assert_eq!(
            CallHooks::transform_response(&hooks, &trace, "output".into(), timing)
                .await
                .unwrap(),
            "output"
        );
        assert!(trace.lock().unwrap().is_empty());
    }
}
