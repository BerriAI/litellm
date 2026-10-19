//! The configured hooks and observers, run for each request.

use std::{sync::Arc, time::SystemTime};

use bytes::Bytes;

use crate::{
    facts::{Facts, Outcome, Record, RequestFacts},
    hook::{Hook, Refusal, RequestEdit, Settle},
    observer::Observers,
    pricing::Prices,
};

pub struct Pipeline {
    hooks: Vec<Arc<dyn Hook>>,
    observers: Observers,
    prices: Arc<Prices>,
    reads: Facts,
}

impl Pipeline {
    /// `hooks` run in the order given, which is the config's.
    pub fn new(hooks: Vec<Arc<dyn Hook>>, observers: Observers, prices: Arc<Prices>) -> Self {
        let mut reads = observers.reads();
        for hook in &hooks {
            reads |= hook.reads();
            hook.prepare(&prices);
        }
        // Cost is computed from usage and the deployment that served the request.
        if reads.contains(Facts::COST) {
            reads |= Facts::USAGE | Facts::DEPLOYMENT;
        }
        Self {
            hooks,
            observers,
            prices,
            reads,
        }
    }

    /// No hooks and no observers: the gateway can skip building facts altogether.
    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty() && self.observers.is_empty()
    }

    /// Whether some hook or observer reads every fact in `facts`.
    pub fn reads(&self, facts: Facts) -> bool {
        self.reads.contains(facts)
    }

    pub fn has_hooks(&self) -> bool {
        !self.hooks.is_empty()
    }

    pub fn observers(&self) -> &Observers {
        &self.observers
    }

    pub fn prices(&self) -> &Prices {
        &self.prices
    }

    /// Runs every hook in order. When one refuses, the hooks before it are told the request
    /// ended with that refusal, and so are the observers.
    pub async fn admit(&self, request: Arc<RequestFacts>) -> Result<Admission, Refusal> {
        let admitted = self.run_hooks(request).await;
        // Refused or not, the request's facts outlive this call (in observer records, or for
        // the whole response), and should not keep the connection's read buffer with them.
        let request = match &admitted {
            Ok(admission) => &admission.request,
            Err((request, _)) => request,
        };
        request.release_body(self.observers.reads().contains(Facts::REQUEST_BODY));
        admitted.map_err(|(_, refusal)| refusal)
    }

    async fn run_hooks(
        &self,
        request: Arc<RequestFacts>,
    ) -> Result<Admission, (Arc<RequestFacts>, Refusal)> {
        let mut admission = Admission {
            request,
            edit: RequestEdit::default(),
            settles: Vec::new(),
        };
        for hook in &self.hooks {
            admission.edit.scope(hook.name(), hook.writes());
            match hook.run(&admission.request, &mut admission.edit).await {
                Ok(Some(settle)) => admission.settles.push(settle),
                Ok(None) => {}
                Err(refusal) => {
                    let outcome = Outcome {
                        deployment: None,
                        status: refusal.status,
                        usage: None,
                        cost: None,
                        time_to_first_byte: None,
                        upstream_started_at: None,
                        ended_at: SystemTime::now(),
                        interrupted: false,
                    };
                    let request = Arc::clone(&admission.request);
                    self.finish(admission, outcome, None);
                    return Err((request, refusal));
                }
            }
        }
        Ok(admission)
    }

    /// Records how the request ended: prices its usage, lets hooks settle, and hands the
    /// record to the observers.
    pub fn finish(&self, admission: Admission, mut outcome: Outcome, response_body: Option<Bytes>) {
        let Admission {
            request,
            mut edit,
            settles,
        } = admission;
        if outcome.cost.is_none()
            && let (Some(usage), Some(deployment)) = (&outcome.usage, &outcome.deployment)
        {
            outcome.cost = self.prices.cost(&deployment.id, usage);
        }
        let record = Record {
            request,
            outcome,
            annotations: edit.take_annotations(),
            response_body: response_body.filter(|_| self.reads(Facts::RESPONSE_BODY)),
        };
        for settle in settles {
            settle.settle(&record);
        }
        if !self.observers.is_empty() {
            self.observers.send(Arc::new(record));
        }
    }

    /// Lets observers finish what they hold. Call once the server has stopped taking requests.
    pub async fn close(&self) {
        self.observers.close().await;
    }
}

/// A request the hooks let through, with the changes they asked for.
pub struct Admission {
    request: Arc<RequestFacts>,
    edit: RequestEdit,
    settles: Vec<Box<dyn Settle>>,
}

impl Admission {
    pub fn request(&self) -> &Arc<RequestFacts> {
        &self.request
    }

    pub fn edit(&self) -> &RequestEdit {
        &self.edit
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use futures_util::future::BoxFuture;
    use http::HeaderMap;
    use serde_json::{Value, json};

    use super::*;
    use crate::{
        facts::{DeploymentFacts, Route},
        hook::Targets,
        observer::Observer,
        usage::Usage,
    };

    fn request() -> Arc<RequestFacts> {
        Arc::new(RequestFacts::new(
            "r".into(),
            Route::Messages,
            None,
            None,
            "m".into(),
            Arc::new(HeaderMap::new()),
            false,
            None,
            Bytes::from_static(b"{}"),
        ))
    }

    fn outcome() -> Outcome {
        Outcome {
            deployment: Some(DeploymentFacts {
                id: "d".into(),
                ..DeploymentFacts::default()
            }),
            status: 200,
            usage: Some(Usage {
                input_tokens: 10,
                output_tokens: 5,
                ..Usage::default()
            }),
            cost: None,
            time_to_first_byte: None,
            upstream_started_at: None,
            ended_at: SystemTime::now(),
            interrupted: false,
        }
    }

    type Log = Arc<Mutex<Vec<String>>>;

    struct Step {
        name: &'static str,
        refuse: bool,
        log: Log,
    }

    struct Settled(&'static str, Log);

    impl Settle for Settled {
        fn settle(self: Box<Self>, record: &Record) {
            self.1
                .lock()
                .unwrap()
                .push(format!("settle {} {}", self.0, record.outcome.status));
        }
    }

    impl Hook for Step {
        fn name(&self) -> &'static str {
            self.name
        }

        fn reads(&self) -> Facts {
            Facts::KEY
        }

        fn writes(&self) -> Targets {
            Targets::ADMISSION | Targets::ANNOTATIONS
        }

        fn run<'a>(
            &'a self,
            _request: &'a Arc<RequestFacts>,
            edit: &'a mut RequestEdit,
        ) -> BoxFuture<'a, Result<Option<Box<dyn Settle>>, Refusal>> {
            Box::pin(async move {
                self.log.lock().unwrap().push(format!("run {}", self.name));
                if self.refuse {
                    return Err(Refusal::new(429, "rate_limit", "slow down"));
                }
                edit.annotate("last", Value::from(self.name));
                Ok(Some(
                    Box::new(Settled(self.name, self.log.clone())) as Box<dyn Settle>
                ))
            })
        }
    }

    struct Costs(Mutex<Vec<(Option<f64>, Option<Value>)>>);

    impl Observer for Costs {
        fn name(&self) -> &'static str {
            "costs"
        }

        fn reads(&self) -> Facts {
            Facts::COST | Facts::ANNOTATIONS
        }

        fn observe<'a>(&'a self, record: &'a Record) -> BoxFuture<'a, ()> {
            Box::pin(async move {
                self.0
                    .lock()
                    .unwrap()
                    .push((record.outcome.cost, record.annotations.get("last").cloned()));
            })
        }
    }

    fn step(name: &'static str, refuse: bool, log: &Log) -> Arc<dyn Hook> {
        Arc::new(Step {
            name,
            refuse,
            log: log.clone(),
        })
    }

    #[tokio::test]
    async fn hooks_run_in_order_and_settle_with_the_cost() {
        let log = Log::default();
        let costs = Arc::new(Costs(Mutex::default()));
        let mut prices = Prices::default();
        prices.insert(
            "d".into(),
            json!({"input_cost_per_token": 1.0, "output_cost_per_token": 2.0})
                .as_object()
                .unwrap(),
        );
        let pipeline = Pipeline::new(
            vec![step("a", false, &log), step("b", false, &log)],
            Observers::start(vec![costs.clone()], 8),
            Arc::new(prices),
        );
        assert!(pipeline.reads(Facts::USAGE | Facts::DEPLOYMENT | Facts::KEY));
        assert!(!pipeline.reads(Facts::RESPONSE_BODY));

        let admission = pipeline.admit(request()).await.unwrap();
        pipeline.finish(admission, outcome(), None);
        pipeline.close().await;

        assert_eq!(
            *log.lock().unwrap(),
            ["run a", "run b", "settle a 200", "settle b 200"]
        );
        assert_eq!(
            *costs.0.lock().unwrap(),
            [(Some(20.0), Some(Value::from("b")))]
        );
    }

    struct Bodies;

    impl Observer for Bodies {
        fn name(&self) -> &'static str {
            "bodies"
        }

        fn reads(&self) -> Facts {
            Facts::REQUEST_BODY
        }

        fn observe<'a>(&'a self, _record: &'a Record) -> BoxFuture<'a, ()> {
            Box::pin(async {})
        }
    }

    /// Hooks see the request as sent; afterwards the facts hold no slice of the read buffer,
    /// only a copy when an observer reads the body.
    #[rstest::rstest]
    #[case::no_observer_reads_it(Vec::new(), b"")]
    #[case::an_observer_reads_it(vec![Arc::new(Bodies) as Arc<dyn Observer>], b"{\"model\":\"m\"}")]
    #[tokio::test]
    async fn the_raw_body_is_released_after_the_hooks(
        #[case] observers: Vec<Arc<dyn Observer>>,
        #[case] kept: &[u8],
    ) {
        let buffer = Bytes::from_static(b"{\"model\":\"m\"} and the rest of the read buffer");
        let raw = buffer.slice(..13);
        let request = Arc::new(RequestFacts::new(
            "r".into(),
            Route::Messages,
            None,
            None,
            "m".into(),
            Arc::new(HeaderMap::new()),
            false,
            None,
            raw.clone(),
        ));
        assert_eq!(request.body().as_ptr(), raw.as_ptr());
        let pipeline = Pipeline::new(Vec::new(), Observers::start(observers, 8), Arc::default());
        let admission = pipeline.admit(request).await.unwrap();

        let body = admission.request().body();
        assert_eq!(&body[..], kept);
        assert!(body.is_empty() || body.as_ptr() != raw.as_ptr());
        assert_eq!(admission.request().input_tokens(), 4);
    }

    #[tokio::test]
    async fn a_refusal_settles_the_hooks_before_it() {
        let log = Log::default();
        let pipeline = Pipeline::new(
            vec![
                step("a", false, &log),
                step("limit", true, &log),
                step("c", false, &log),
            ],
            Observers::start(Vec::new(), 8),
            Arc::default(),
        );
        let refusal = pipeline.admit(request()).await.err().unwrap();
        assert_eq!(refusal.status, 429);
        assert_eq!(*log.lock().unwrap(), ["run a", "run limit", "settle a 429"]);
    }
}
