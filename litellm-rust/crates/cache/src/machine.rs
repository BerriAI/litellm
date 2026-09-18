use std::time::Duration;

use litellm_callbacks::event::epoch_seconds;
use litellm_callbacks::host::HostResult;
use litellm_callbacks::layer::Layer;
use litellm_callbacks::machine::{HostFailure, Interrupted, Machine, MachineStep, Step};
use litellm_callbacks::route::Route;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::base_cache::CacheKwargs;
use crate::caching::{CacheBackend, CacheControls, CacheEntry};

/// Serves a call from the cache or stores its result, around one machine.
///
/// Sits inside the router, around each attempt, which is where Python's `@client` put it:
/// a hit completes the attempt without a provider call, a miss runs it and stores on
/// success. Cache errors never fail the call; they degrade to a miss or a skipped store.
pub struct Cached<M: Machine> {
    inner: M,
    backend: CacheBackend,
    key: String,
    controls: CacheControls,
    kwargs: CacheKwargs,
    max_age: Option<Duration>,
    state: State,
}

enum State {
    Lookup,
    Running,
    Done,
}

impl<M> Cached<M>
where
    M: Machine,
    M::Complete: Serialize + DeserializeOwned,
{
    fn lookup(&self) -> impl std::future::Future<Output = Option<M::Complete>> + Send {
        let reads = self.controls.reads();
        let (backend, key, kwargs, max_age) = (
            self.backend.clone(),
            self.key.clone(),
            self.kwargs.clone(),
            self.max_age,
        );
        async move {
            if !reads {
                return None;
            }
            let entry = fetch(&backend, &key, &kwargs).await?;
            if !entry.fresh(Duration::from_secs_f64(epoch_seconds()), max_age) {
                return None;
            }
            serde_json::from_value(entry.response).ok()
        }
    }

    fn store(&self, complete: &M::Complete) -> impl std::future::Future<Output = ()> + Send {
        let response = self
            .controls
            .writes()
            .then(|| serde_json::to_value(complete).ok())
            .flatten();
        let (backend, key, kwargs) = (self.backend.clone(), self.key.clone(), self.kwargs.clone());
        async move {
            let Some(response) = response else {
                return;
            };
            let entry = CacheEntry {
                timestamp: epoch_seconds(),
                response,
            };
            put(&backend, &key, entry, kwargs).await;
        }
    }

    async fn step(
        &mut self,
        result: Option<HostResult<M::Route>>,
    ) -> Result<MachineStep<M::Route, M::Complete>, <M::Route as Route>::Error> {
        loop {
            match std::mem::replace(&mut self.state, State::Done) {
                State::Lookup => {
                    if let Some(hit) = self.lookup().await {
                        return Ok(MachineStep::Complete(hit));
                    }
                    self.state = State::Running;
                }
                State::Running => {
                    return match self.inner.resume(result).await {
                        Ok(MachineStep::Host(op)) => {
                            self.state = State::Running;
                            Ok(MachineStep::Host(op))
                        }
                        Ok(MachineStep::Complete(complete)) => {
                            self.store(&complete).await;
                            Ok(MachineStep::Complete(complete))
                        }
                        Err(error) => Err(error),
                    };
                }
                State::Done => unreachable!("cached machine resumed after completion"),
            }
        }
    }
}

async fn fetch(backend: &CacheBackend, key: &str, kwargs: &CacheKwargs) -> Option<CacheEntry> {
    backend.async_get_cache(key, kwargs).await.ok().flatten()
}

async fn put(backend: &CacheBackend, key: &str, entry: CacheEntry, kwargs: CacheKwargs) {
    let _ = backend.async_set_cache(key, entry, kwargs).await;
}

impl<M> Machine for Cached<M>
where
    M: Machine,
    M::Complete: Serialize + DeserializeOwned,
{
    type Route = M::Route;
    type Complete = M::Complete;

    fn resume(&mut self, result: Option<HostResult<Self::Route>>) -> Step<'_, Self> {
        Box::pin(self.step(result))
    }

    fn interrupt(
        &mut self,
        failure: HostFailure<<Self::Route as Route>::Error>,
    ) -> Interrupted<'_, Self> {
        self.state = State::Done;
        self.inner.interrupt(failure)
    }
}

/// Per-call cache configuration: the key is computed by the host from the typed request
/// (`cache_key`), the controls from the call's cache kwargs and the global setting.
#[derive(Clone)]
pub struct CacheLayer {
    backend: CacheBackend,
    key: String,
    controls: CacheControls,
    kwargs: CacheKwargs,
    max_age: Option<Duration>,
}

impl CacheLayer {
    pub fn new(backend: CacheBackend, key: String, controls: CacheControls) -> Self {
        Self {
            backend,
            key,
            controls,
            kwargs: CacheKwargs::default(),
            max_age: None,
        }
    }

    pub fn with_kwargs(self, kwargs: CacheKwargs) -> Self {
        Self { kwargs, ..self }
    }

    pub fn with_max_age(self, max_age: Option<Duration>) -> Self {
        Self { max_age, ..self }
    }
}

impl<M> Layer<M> for CacheLayer
where
    M: Machine,
    M::Complete: Serialize + DeserializeOwned,
{
    type Output = Cached<M>;

    fn layer(&self, inner: M) -> Cached<M> {
        Cached {
            inner,
            backend: self.backend.clone(),
            key: self.key.clone(),
            controls: self.controls,
            kwargs: self.kwargs.clone(),
            max_age: self.max_age,
            state: State::Lookup,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    use litellm_callbacks::host::HostOp;
    use litellm_callbacks::layer::Stack;

    use super::*;
    use crate::BaseCache;

    #[derive(Default)]
    struct MemoryBackend {
        entries: Mutex<HashMap<String, CacheEntry>>,
        gets: Mutex<u32>,
    }

    impl BaseCache for MemoryBackend {
        type Value = CacheEntry;

        fn set_cache(
            &self,
            key: &str,
            value: CacheEntry,
            _: CacheKwargs,
        ) -> Result<(), crate::Error> {
            self.entries.lock().unwrap().insert(key.into(), value);
            Ok(())
        }

        fn get_cache(
            &self,
            key: &str,
            _: &CacheKwargs,
        ) -> Result<Option<CacheEntry>, crate::Error> {
            *self.gets.lock().unwrap() += 1;
            Ok(self.entries.lock().unwrap().get(key).cloned())
        }

        fn delete_cache(&self, key: &str) -> Result<(), crate::Error> {
            self.entries.lock().unwrap().remove(key);
            Ok(())
        }

        fn flush_cache(&self) -> Result<(), crate::Error> {
            self.entries.lock().unwrap().clear();
            Ok(())
        }

        fn disconnect(&self) -> crate::CacheFuture<'_, ()> {
            Box::pin(async { Ok(()) })
        }

        fn test_connection(&self) -> crate::CacheFuture<'_, crate::CacheConnectionResult> {
            Box::pin(async {
                Ok(crate::CacheConnectionResult {
                    status: crate::CacheConnectionStatus::Success,
                    message: String::new(),
                    error: None,
                })
            })
        }
    }

    struct Unit;

    impl Route for Unit {
        type Response = String;
        type Error = &'static str;
        type Op = &'static str;
        type OpResult = ();
    }

    struct Provider {
        calls: Arc<Mutex<u32>>,
        response: String,
    }

    impl Machine for Provider {
        type Route = Unit;
        type Complete = String;

        fn resume(&mut self, result: Option<HostResult<Unit>>) -> Step<'_, Self> {
            Box::pin(async move {
                if result.is_none() {
                    return Ok(MachineStep::Host(HostOp::Route("send")));
                }
                *self.calls.lock().unwrap() += 1;
                Ok(MachineStep::Complete(self.response.clone()))
            })
        }

        fn interrupt(&mut self, failure: HostFailure<&'static str>) -> Interrupted<'_, Self> {
            Box::pin(async move { Err(failure.into_error()) })
        }
    }

    fn controls(reads: bool, writes: bool) -> CacheControls {
        CacheControls {
            supported_call_type: true,
            configured: true,
            native_backend: true,
            default_on: true,
            caching: None,
            no_cache: !reads,
            no_store: !writes,
            use_cache: false,
        }
    }

    async fn drain<M: Machine<Route = Unit, Complete = String>>(machine: &mut M) -> (u32, String) {
        let mut ops = 0;
        let mut result = None;
        loop {
            match machine.resume(result.take()).await.unwrap() {
                MachineStep::Host(HostOp::Route(_)) => {
                    ops += 1;
                    result = Some(HostResult::Route(()));
                }
                MachineStep::Host(_) => unreachable!(),
                MachineStep::Complete(value) => return (ops, value),
            }
        }
    }

    fn provider(calls: &Arc<Mutex<u32>>, response: &str) -> Provider {
        Provider {
            calls: Arc::clone(calls),
            response: response.into(),
        }
    }

    #[tokio::test]
    async fn miss_runs_the_inner_machine_and_stores_then_hit_skips_it() {
        let backend: CacheBackend = Arc::new(MemoryBackend::default());
        let layer = CacheLayer::new(backend.clone(), "k".into(), controls(true, true));
        let calls = Arc::new(Mutex::new(0));

        let mut first = Stack::new(provider(&calls, "fresh"))
            .layer(layer.clone())
            .build();
        assert_eq!(drain(&mut first).await, (1, "fresh".into()));
        assert_eq!(*calls.lock().unwrap(), 1);

        let mut second = Stack::new(provider(&calls, "unused")).layer(layer).build();
        assert_eq!(drain(&mut second).await, (0, "fresh".into()));
        assert_eq!(*calls.lock().unwrap(), 1);
    }

    #[tokio::test]
    async fn controls_gate_reads_and_writes_independently() {
        let memory = Arc::new(MemoryBackend::default());
        let backend: CacheBackend = memory.clone();
        let calls = Arc::new(Mutex::new(0));

        let no_store = CacheLayer::new(backend.clone(), "k".into(), controls(true, false));
        let mut machine = Stack::new(provider(&calls, "a")).layer(no_store).build();
        drain(&mut machine).await;
        assert!(memory.entries.lock().unwrap().is_empty());

        let store = CacheLayer::new(backend.clone(), "k".into(), controls(true, true));
        let mut machine = Stack::new(provider(&calls, "a")).layer(store).build();
        drain(&mut machine).await;
        let gets_before = *memory.gets.lock().unwrap();

        let no_read = CacheLayer::new(backend, "k".into(), controls(false, true));
        let mut machine = Stack::new(provider(&calls, "b")).layer(no_read).build();
        assert_eq!(drain(&mut machine).await, (1, "b".into()));
        assert_eq!(*memory.gets.lock().unwrap(), gets_before);
    }

    #[tokio::test]
    async fn stale_entries_are_a_miss() {
        let memory = Arc::new(MemoryBackend::default());
        memory.entries.lock().unwrap().insert(
            "k".into(),
            CacheEntry {
                timestamp: epoch_seconds() - 100.0,
                response: serde_json::json!("old"),
            },
        );
        let backend: CacheBackend = memory;
        let calls = Arc::new(Mutex::new(0));
        let layer = CacheLayer::new(backend, "k".into(), controls(true, false))
            .with_max_age(Some(Duration::from_secs(10)));
        let mut machine = Stack::new(provider(&calls, "new")).layer(layer).build();
        assert_eq!(drain(&mut machine).await, (1, "new".into()));
    }
}
