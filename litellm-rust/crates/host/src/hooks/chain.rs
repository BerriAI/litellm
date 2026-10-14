use super::{
    AnyGroup, ArgumentHooks, Cancelled, OutcomeHooks, ProviderHooks, ResponseHooks, Surface,
};
use crate::{
    interceptors::{ExecutionFacts, RawResponse, RequestContext, WireRequest},
    lifecycle::{FailureOrigin, Timing},
};

/// The groups mounted on a call, run as one group of each facet. Request-side stages run
/// in mount order, response-side and terminal stages unwind in reverse, like layers.
pub struct Hooks<S: Surface> {
    groups: Vec<Box<S::Group>>,
}

impl<S: Surface> Hooks<S> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, group: Box<S::Group>) -> Self {
        self.groups.push(group);
        self
    }

    pub fn groups(&self) -> impl Iterator<Item = &S::Group> {
        self.groups.iter().map(Box::as_ref)
    }

    pub fn groups_mut(&mut self) -> impl Iterator<Item = &mut S::Group> {
        self.groups.iter_mut().map(Box::as_mut)
    }
}

impl<S: Surface> Default for Hooks<S> {
    fn default() -> Self {
        Self { groups: Vec::new() }
    }
}

impl<S: Surface> ArgumentHooks<S> for Hooks<S> {
    async fn prepare<'a>(
        &'a self,
        task: S::Task<'a>,
        arguments: S::Arguments,
        started_at: f64,
    ) -> Result<S::Arguments, S::Error> {
        let mut arguments = arguments;
        for group in &self.groups {
            arguments = group.prepare(task, arguments, started_at).await?;
        }
        Ok(arguments)
    }

    fn prepared(&self, arguments: &S::Arguments) -> Result<(), S::Error> {
        self.groups
            .iter()
            .try_for_each(|group| group.prepared(arguments))
    }
}

impl<S: Surface> ProviderHooks<S> for Hooks<S> {
    async fn before_provider_request<'a>(
        &'a self,
        task: S::Task<'a>,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> Result<Box<WireRequest>, S::Error> {
        let mut wire = wire;
        for group in &self.groups {
            wire = group.before_provider_request(task, wire, context).await?;
        }
        Ok(wire)
    }

    async fn after_provider_response<'a>(
        &'a self,
        task: S::Task<'a>,
        raw: &RawResponse,
    ) -> Result<(), S::Error> {
        for group in &self.groups {
            group.after_provider_response(task, raw).await?;
        }
        Ok(())
    }

    async fn result_ready<'a>(
        &'a self,
        task: S::Task<'a>,
        facts: &ExecutionFacts,
    ) -> Result<(), S::Error> {
        for group in &self.groups {
            group.result_ready(task, facts).await?;
        }
        Ok(())
    }
}

impl<S: Surface> ResponseHooks<S> for Hooks<S> {
    async fn transform<'a>(
        &'a self,
        task: S::Task<'a>,
        response: S::Response,
        timing: Timing,
    ) -> Result<S::Response, S::Error> {
        let mut response = response;
        for group in self.groups.iter().rev() {
            response = group.transform(task, response, timing).await?;
        }
        Ok(response)
    }

    fn stream_opened(&self, head: &S::Response) -> Result<(), S::Error> {
        self.groups
            .iter()
            .rev()
            .try_for_each(|group| group.stream_opened(head))
    }

    fn stream_chunk(&self, chunk: &S::Chunk) -> Result<(), S::Error> {
        self.groups
            .iter()
            .rev()
            .try_for_each(|group| group.stream_chunk(chunk))
    }
}

impl<S: Surface> OutcomeHooks<S> for Hooks<S> {
    async fn started<'a>(
        &'a self,
        task: S::Task<'a>,
        start_time: f64,
    ) -> Result<(), Cancelled<S::Error>> {
        for group in &self.groups {
            group.started(task, start_time).await?;
        }
        Ok(())
    }

    async fn succeeded<'a>(
        &'a self,
        task: S::Task<'a>,
        timing: Timing,
        response: &S::Response,
    ) -> Result<(), Cancelled<S::Error>> {
        for group in self.groups.iter().rev() {
            group.succeeded(task, timing, response).await?;
        }
        Ok(())
    }

    async fn failed<'a>(
        &'a self,
        task: S::Task<'a>,
        timing: Timing,
        origin: FailureOrigin,
        error: &S::Failure,
    ) -> Result<(), Cancelled<S::Error>> {
        for group in self.groups.iter().rev() {
            group.failed(task, timing, origin, error).await?;
        }
        Ok(())
    }

    async fn cancelled<'a>(
        &'a self,
        task: S::Task<'a>,
        timing: Timing,
    ) -> Result<(), Cancelled<S::Error>> {
        for group in self.groups.iter().rev() {
            group.cancelled(task, timing).await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use rstest::{fixture, rstest};

    use super::*;

    struct TestSurface;

    impl Surface for TestSurface {
        type Task<'a> = &'a Mutex<Vec<String>>;
        type Arguments = String;
        type Response = String;
        type Chunk = u32;
        type Error = &'static str;
        type Failure = &'static str;
        type Group = dyn AnyGroup<Self>;
    }

    /// A group that tags every value it transforms with its name, logs every stage it
    /// runs, and fails one stage when told to.
    struct Group {
        name: &'static str,
        suspensions: u8,
        reject: Option<&'static str>,
    }

    impl Group {
        fn mounted(self) -> Box<dyn AnyGroup<TestSurface>> {
            Box::new(self)
        }

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

    impl ArgumentHooks<TestSurface> for Group {
        async fn prepare<'a>(
            &'a self,
            trace: &'a Mutex<Vec<String>>,
            arguments: String,
            _: f64,
        ) -> Result<String, &'static str> {
            self.value(trace, "prepare", arguments).await
        }
    }

    impl ProviderHooks<TestSurface> for Group {
        async fn before_provider_request<'a>(
            &'a self,
            trace: &'a Mutex<Vec<String>>,
            wire: Box<WireRequest>,
            _: &RequestContext,
        ) -> Result<Box<WireRequest>, &'static str> {
            let url = self.value(trace, "wire", wire.url.clone()).await?;
            Ok(Box::new(WireRequest { url, ..*wire }))
        }
    }

    impl ResponseHooks<TestSurface> for Group {
        async fn transform<'a>(
            &'a self,
            trace: &'a Mutex<Vec<String>>,
            response: String,
            _: Timing,
        ) -> Result<String, &'static str> {
            self.value(trace, "response", response).await
        }
    }

    impl OutcomeHooks<TestSurface> for Group {
        async fn started<'a>(
            &'a self,
            trace: &'a Mutex<Vec<String>>,
            _: f64,
        ) -> Result<(), Cancelled<&'static str>> {
            self.value(trace, "started", String::new())
                .await
                .map(drop)
                .map_err(Cancelled)
        }

        async fn succeeded<'a>(
            &'a self,
            trace: &'a Mutex<Vec<String>>,
            _: Timing,
            response: &String,
        ) -> Result<(), Cancelled<&'static str>> {
            self.value(trace, response, String::new())
                .await
                .map(drop)
                .map_err(Cancelled)
        }

        async fn failed<'a>(
            &'a self,
            trace: &'a Mutex<Vec<String>>,
            _: Timing,
            _: FailureOrigin,
            error: &&'static str,
        ) -> Result<(), Cancelled<&'static str>> {
            self.value(trace, error, String::new())
                .await
                .map(drop)
                .map_err(Cancelled)
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

    fn chain(
        suspensions: u8,
        rejecting: Option<(&'static str, &'static str)>,
    ) -> Hooks<TestSurface> {
        ["a", "b", "c"]
            .into_iter()
            .fold(Hooks::new(), |hooks, name| {
                hooks.with(
                    Group {
                        name,
                        suspensions,
                        reject: rejecting
                            .filter(|(group, _)| *group == name)
                            .map(|(_, error)| error),
                    }
                    .mounted(),
                )
            })
    }

    #[rstest]
    #[tokio::test]
    async fn request_stages_run_in_mount_order_and_response_stages_unwind(
        trace: Mutex<Vec<String>>,
        timing: Timing,
        context: RequestContext,
        #[values(0, 1, 2)] suspensions: u8,
    ) {
        let hooks = chain(suspensions, None);
        OutcomeHooks::started(&hooks, &trace, 0.0).await.unwrap();
        let arguments = ArgumentHooks::prepare(&hooks, &trace, "input".into(), 0.0)
            .await
            .unwrap();
        assert_eq!(arguments, "inputabc");
        let wire = Box::new(WireRequest {
            url: "url".into(),
            headers: Vec::new(),
            body: serde_json::Value::Null,
        });
        let rewritten = ProviderHooks::before_provider_request(&hooks, &trace, wire, &context)
            .await
            .unwrap();
        assert_eq!(rewritten.url, "urlabc");
        let response = ResponseHooks::transform(&hooks, &trace, "output".into(), timing)
            .await
            .unwrap();
        assert_eq!(response, "outputcba");
        OutcomeHooks::succeeded(&hooks, &trace, timing, &response)
            .await
            .unwrap();
        assert_eq!(
            *trace.lock().unwrap(),
            [
                "started:a",
                "started:b",
                "started:c",
                "prepare:a",
                "prepare:b",
                "prepare:c",
                "wire:a",
                "wire:b",
                "wire:c",
                "response:c",
                "response:b",
                "response:a",
                "outputcba:c",
                "outputcba:b",
                "outputcba:a",
            ]
        );
    }

    #[rstest]
    #[tokio::test]
    async fn a_transformation_error_stops_later_groups(
        trace: Mutex<Vec<String>>,
        #[values(0, 2)] suspensions: u8,
    ) {
        let hooks = chain(suspensions, Some(("b", "rejected")));
        let result = ArgumentHooks::prepare(&hooks, &trace, String::new(), 0.0).await;
        assert_eq!(result, Err("rejected"));
        assert_eq!(*trace.lock().unwrap(), ["prepare:a", "prepare:b"]);
    }

    #[rstest]
    #[case::success(true)]
    #[case::failure(false)]
    #[tokio::test]
    async fn a_cancelled_outcome_stops_the_groups_mounted_before_it(
        trace: Mutex<Vec<String>>,
        timing: Timing,
        #[values(0, 2)] suspensions: u8,
        #[case] succeeded: bool,
    ) {
        let hooks = chain(suspensions, Some(("b", "cancelled")));
        let response = "selected".to_owned();
        let result = if succeeded {
            OutcomeHooks::succeeded(&hooks, &trace, timing, &response).await
        } else {
            OutcomeHooks::failed(&hooks, &trace, timing, FailureOrigin::Host, &"failed").await
        };
        assert_eq!(result, Err(Cancelled("cancelled")));
        let reached = if succeeded {
            ["selected:c", "selected:b"]
        } else {
            ["failed:c", "failed:b"]
        };
        assert_eq!(*trace.lock().unwrap(), reached);
    }

    #[rstest]
    #[tokio::test]
    async fn an_empty_chain_preserves_values(trace: Mutex<Vec<String>>, timing: Timing) {
        let hooks = Hooks::<TestSurface>::new();
        assert_eq!(
            ArgumentHooks::prepare(&hooks, &trace, "input".into(), 0.0)
                .await
                .unwrap(),
            "input"
        );
        assert_eq!(
            ResponseHooks::transform(&hooks, &trace, "output".into(), timing)
                .await
                .unwrap(),
            "output"
        );
        assert!(trace.lock().unwrap().is_empty());
    }
}
