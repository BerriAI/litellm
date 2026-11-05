use std::{
    future::Future,
    sync::{Arc, Mutex},
    time::Duration,
};

use futures_util::future::{BoxFuture, join_all};
use litellm_redis_client::{Declared, RedisBatch, RedisClient};
use tokio::{runtime::Handle, sync::oneshot, task::JoinHandle};

use crate::memory::lock;

/// One request's Redis batches: a pre-call batch for the reads it needs before answering and a
/// post-call batch for the writes nobody waits on, one of each per Redis client. The host that owns
/// the request builds one and passes it down; it knows nothing else about the request.
#[derive(Clone)]
pub struct RequestRedis(Arc<Shared>);

struct Shared {
    runtime: Handle,
    post_call_deadline: Duration,
    batches: Mutex<Vec<ClientBatches>>,
    follow_ups: Mutex<Vec<JoinHandle<()>>>,
    deadline: Mutex<Option<oneshot::Sender<()>>>,
}

#[derive(Clone)]
struct ClientBatches {
    client: RedisClient,
    pre: RedisBatch,
    post: RedisBatch,
}

impl RequestRedis {
    /// `runtime` runs the post-call deadline, so declarations may come from any thread, including a
    /// Python one that has no Tokio context.
    pub fn new(runtime: Handle, post_call_deadline: Duration) -> Self {
        Self(Arc::new(Shared {
            runtime,
            post_call_deadline,
            batches: Mutex::new(Vec::new()),
            follow_ups: Mutex::new(Vec::new()),
            deadline: Mutex::new(None),
        }))
    }

    pub fn pre(&self, client: &RedisClient) -> RedisBatch {
        self.batches_for(client).pre
    }

    /// The post-call batch, which goes out on `flush_post` or `post_call_deadline` after the first
    /// declaration, whichever comes first.
    pub fn post(&self, client: &RedisClient) -> RedisBatch {
        self.arm_deadline();
        self.batches_for(client).post
    }

    /// Runs work that follows this request's post-call batch, such as mirroring settled totals
    /// into memory, on the request's runtime, from any thread. `flush_post` waits for it, so the
    /// shutdown drain does not drop a fallback write still in flight.
    pub fn spawn(&self, task: impl Future<Output = ()> + std::marker::Send + 'static) {
        let follow_up = self.0.runtime.spawn(task);
        lock(&self.0.follow_ups).push(follow_up);
    }

    pub async fn flush_pre(&self) {
        join_all(self.batches().iter().map(|batches| batches.pre.flush())).await;
    }

    /// Sends every post-call op declared so far and waits for the follow-ups spawned on them; the
    /// shutdown drain calls it too.
    pub async fn flush_post(&self) {
        lock(&self.0.deadline).take();
        let follow_ups = std::mem::take(&mut *lock(&self.0.follow_ups));
        join_all(self.batches().iter().map(|batches| batches.post.flush())).await;
        join_all(follow_ups)
            .await
            .into_iter()
            .filter_map(Result::err)
            .for_each(|error| tracing::warn!(%error, "a post-call follow-up failed"));
    }

    fn batches(&self) -> Vec<ClientBatches> {
        lock(&self.0.batches).clone()
    }

    fn batches_for(&self, client: &RedisClient) -> ClientBatches {
        let mut batches = lock(&self.0.batches);
        if let Some(existing) = batches
            .iter()
            .find(|batches| batches.client.same_client(client))
        {
            return existing.clone();
        }
        let created = ClientBatches {
            client: client.clone(),
            pre: client.batch(),
            post: client.batch(),
        };
        batches.push(created.clone());
        created
    }

    fn arm_deadline(&self) {
        let mut deadline = lock(&self.0.deadline);
        if deadline.is_some() {
            return;
        }
        let (disarm, disarmed) = oneshot::channel();
        *deadline = Some(disarm);
        let request = self.clone();
        self.0.runtime.spawn(async move {
            tokio::select! {
                () = tokio::time::sleep(request.0.post_call_deadline) => request.flush_post().await,
                _ = disarmed => {}
            }
        });
    }
}

/// Which batch an op rides, picked by the caller because only it knows when it needs the answer.
/// The store owns its client, so it calls `batch`, the one place a request and a client meet.
#[derive(Clone, Copy)]
pub enum Send<'a> {
    /// On its own, sent at the first `resolve`.
    Now,
    /// The request's pre-call batch, sent at the first `resolve` or `flush_pre`.
    PreCall(&'a RequestRedis),
    /// The request's post-call batch, sent at `flush_post` or the post-call deadline.
    PostCall(&'a RequestRedis),
}

impl Send<'_> {
    pub fn batch(&self, client: &RedisClient) -> RedisBatch {
        match self {
            Self::Now => RedisBatch::alone(client.clone()),
            Self::PreCall(request) => request.pre(client),
            Self::PostCall(request) => request.post(client),
        }
    }
}

/// A store op declared on the batch its `Send` picked: building it sends nothing, and awaiting it
/// flushes that batch if nobody has yet.
pub type Pending<T> = BoxFuture<'static, Result<T, crate::Error>>;

pub(crate) fn pending<T: std::marker::Send + 'static>(declared: Declared<T>) -> Pending<T> {
    Box::pin(async move { Ok(declared.resolve().await?) })
}

/// An op answered without Redis, from memory.
pub(crate) fn ready<T: std::marker::Send + 'static>(value: T) -> Pending<T> {
    Box::pin(std::future::ready(Ok(value)))
}
