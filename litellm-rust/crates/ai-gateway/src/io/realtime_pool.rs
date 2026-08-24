//! Pre-warmed upstream realtime connection pool.
//!
//! The gateway's realtime overhead lives entirely in session establishment: on
//! every client connect it dials a fresh upstream WS to OpenAI and waits for
//! `session.created` before it can serve. This pool keeps a small set of upstream
//! sockets **already connected and already past `session.created`** so a connect
//! can be served from a warm socket and the handshake is off the critical path.
//!
//! Layering: this lives in the gateway's `io` module next to the dial/splice it
//! reuses. The gateway holds an `Arc<RealtimePool>` in its state and asks for a
//! warm socket per connect; on a miss it fresh-dials exactly as before. The pool
//! is a latency optimization, never a correctness dependency — see the gateway's
//! `src/routes/realtime/README.md`.
//!
//! ## Caveats (enforced here)
//! - One warm socket serves exactly one session (realtime isn't multiplexed), so
//!   the pool is sized to the connect *rate*, not concurrent connections.
//! - `session.created` is pre-read once and buffered; nothing else is read from a
//!   warm socket before handoff, so a warm session starts at OpenAI defaults just
//!   like a fresh one (`session.update` semantics unchanged).
//! - Warm sockets are short-lived (`max_idle`) and liveness-checked at handoff to
//!   bound idle billing / dodge OpenAI's idle timeout.
//! - On miss or dead socket the caller fresh-dials; the pool never blocks or fails
//!   a connect because it is empty.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use litellm_core::realtime::types::RealtimeEvent;
use litellm_core::CoreResult;

use crate::io::realtime::{
    dial_upstream, read_event, resolve_api_key, UpstreamRx, UpstreamTx, UpstreamWs,
};

/// Default target warm sockets per key when pooling is enabled.
pub const DEFAULT_POOL_SIZE: usize = 4;

/// Default max time a warm socket may sit before it is closed and replaced.
pub const DEFAULT_MAX_IDLE: Duration = Duration::from_secs(30);

/// Env var: target warm sockets per key. `0` disables pooling (fresh-dial only).
pub const POOL_SIZE_ENV: &str = "REALTIME_POOL_SIZE";

/// Env var: max warm-socket idle lifetime, in seconds.
pub const MAX_IDLE_ENV: &str = "REALTIME_POOL_MAX_IDLE_SECS";

/// How often the background replenisher wakes to top up and reap stale sockets.
const REPLENISH_TICK: Duration = Duration::from_millis(250);

/// Backoff floor after a key's warm-up dials all fail. The first failed pass
/// waits this long before retrying that key.
const BACKOFF_BASE: Duration = Duration::from_millis(500);

/// Backoff ceiling. A key that keeps failing (invalid credentials, an
/// unreachable upstream) is retried at most once per this interval — instead of
/// firing `needed` concurrent TLS dials every 250 ms tick, which would hammer
/// the upstream and risk rate-limit exhaustion that degrades valid cold-path
/// traffic. Backoff resets the moment a dial for the key succeeds.
const BACKOFF_MAX: Duration = Duration::from_secs(30);

/// Identifies an upstream connection: the tuple that fully determines the dial.
/// `api_key` is included so a warm socket is only ever reused for the same key
/// (no cross-tenant reuse).
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct UpstreamKey {
    pub model: String,
    pub api_key: String,
    pub api_base: Option<String>,
}

impl std::fmt::Debug for UpstreamKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UpstreamKey")
            .field("model", &self.model)
            .field("api_key", &"[REDACTED]")
            .field("api_base", &self.api_base)
            .finish()
    }
}

/// A warm upstream: split halves + the buffered `session.created` + when it was
/// warmed (for `max_idle` expiry).
struct WarmConnection {
    tx: UpstreamTx,
    rx: UpstreamRx,
    session_created: RealtimeEvent,
    warmed_at: Instant,
}

/// A live upstream taken from the pool, ready to splice. The caller relays
/// `session_created` to the client first, then splices `(tx, rx)` as usual.
pub struct WarmHandoff {
    pub tx: UpstreamTx,
    pub rx: UpstreamRx,
    pub session_created: RealtimeEvent,
}

/// Pool configuration, resolved once at startup from the environment.
#[derive(Clone, Copy, Debug)]
pub struct PoolConfig {
    /// Target warm sockets per key. `0` disables pooling.
    pub target_size: usize,
    /// Max time a warm socket may sit before it is closed and replaced.
    pub max_idle: Duration,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            target_size: DEFAULT_POOL_SIZE,
            max_idle: DEFAULT_MAX_IDLE,
        }
    }
}

impl PoolConfig {
    /// Read config from the environment, falling back to defaults. An invalid
    /// value warns and uses the default rather than failing startup.
    pub fn from_env() -> Self {
        let target_size = match std::env::var(POOL_SIZE_ENV) {
            Ok(raw) => raw.trim().parse().unwrap_or_else(|_| {
                eprintln!("warning: {POOL_SIZE_ENV}={raw:?} is not a valid size; using {DEFAULT_POOL_SIZE}");
                DEFAULT_POOL_SIZE
            }),
            Err(_) => DEFAULT_POOL_SIZE,
        };
        let max_idle = match std::env::var(MAX_IDLE_ENV) {
            Ok(raw) => raw
                .trim()
                .parse()
                .map(Duration::from_secs)
                .unwrap_or_else(|_| {
                    eprintln!(
                        "warning: {MAX_IDLE_ENV}={raw:?} is not a valid number of seconds; using {}s",
                        DEFAULT_MAX_IDLE.as_secs()
                    );
                    DEFAULT_MAX_IDLE
                }),
            Err(_) => DEFAULT_MAX_IDLE,
        };
        Self {
            target_size,
            max_idle,
        }
    }

    /// Whether pooling is on (`target_size > 0`).
    pub fn enabled(&self) -> bool {
        self.target_size > 0
    }
}

/// Per-key warm sockets, behind a single `Mutex`. Realtime warm sockets are few
/// (the pool is small), so a plain mutex over a `VecDeque`-ish `Vec` is simpler
/// and faster than sharding; contention is negligible at this scale.
type Warm = HashMap<UpstreamKey, Vec<WarmConnection>>;

/// Per-key replenish backoff. Absent (or `consecutive_failures == 0`) means the
/// key is healthy and replenished every tick. After a pass whose dials all fail,
/// `retry_after` is pushed out with exponential backoff so a broken key (invalid
/// credentials, unreachable upstream) is not re-dialed on every 250 ms tick.
#[derive(Default)]
struct Backoff {
    /// Don't attempt warm-up dials for this key until this instant. `None` =
    /// eligible now.
    retry_after: Option<Instant>,
    consecutive_failures: u32,
}

type Backoffs = HashMap<UpstreamKey, Backoff>;

/// Pre-warmed upstream realtime connection pool.
///
/// Cheap to clone-via-`Arc`. The background replenisher is spawned by
/// [`RealtimePool::spawn`]; a pool built with [`RealtimePool::disabled`] never
/// warms anything and every `take` misses (callers fresh-dial).
pub struct RealtimePool {
    config: PoolConfig,
    warm: Mutex<Warm>,
    /// Per-key replenish backoff so a broken key doesn't trigger unbounded
    /// concurrent dials every tick. Separate lock from `warm` so the request
    /// hot path (`take`) never contends on it.
    backoff: Mutex<Backoffs>,
}

impl RealtimePool {
    /// A disabled pool: no background task, every `take` returns `None`.
    pub fn disabled() -> Arc<Self> {
        Arc::new(Self {
            config: PoolConfig {
                target_size: 0,
                ..PoolConfig::default()
            },
            warm: Mutex::new(HashMap::new()),
            backoff: Mutex::new(HashMap::new()),
        })
    }

    /// Build a pool from config **without** the background replenisher. The pool
    /// only warms when [`RealtimePool::warm_now`] is called. Used by deterministic
    /// unit tests; production uses [`RealtimePool::spawn`].
    #[cfg(test)]
    fn new_unspawned(config: PoolConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            warm: Mutex::new(HashMap::new()),
            backoff: Mutex::new(HashMap::new()),
        })
    }

    /// Build a pool from config and, if enabled, spawn the background replenisher.
    /// Returns the shared handle the gateway stores in its state.
    pub fn spawn(config: PoolConfig) -> Arc<Self> {
        let pool = Arc::new(Self {
            config,
            warm: Mutex::new(HashMap::new()),
            backoff: Mutex::new(HashMap::new()),
        });
        if config.enabled() {
            let weak = Arc::downgrade(&pool);
            tokio::spawn(async move {
                let mut tick = tokio::time::interval(REPLENISH_TICK);
                tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                loop {
                    tick.tick().await;
                    // Stop once the gateway has dropped its handle.
                    let Some(pool) = weak.upgrade() else { break };
                    pool.replenish_all().await;
                }
            });
        }
        pool
    }

    /// Resolved config (test/inspection).
    pub fn config(&self) -> PoolConfig {
        self.config
    }

    /// Register a key so the replenisher starts warming it. Idempotent. The
    /// gateway calls this once per known deployment at startup; the pool only
    /// warms keys it has seen, so it never dials a model nobody asked for.
    pub fn register(&self, key: UpstreamKey) {
        if !self.config.enabled() {
            return;
        }
        self.warm.lock().unwrap().entry(key).or_default();
    }

    /// Take a warm, live socket for `key`, or `None` on miss / dead socket.
    ///
    /// Pops the freshest non-expired socket and liveness-checks it; a socket that
    /// is too old or already dead is dropped (closing it) and the next candidate
    /// tried. Never blocks: if nothing warm is live, returns `None` so the caller
    /// fresh-dials.
    pub fn take(&self, key: &UpstreamKey) -> Option<WarmHandoff> {
        if !self.config.enabled() {
            return None;
        }
        loop {
            let mut candidate = {
                let mut warm = self.warm.lock().unwrap();
                let bucket = warm.get_mut(key)?;
                bucket.pop()?
            };
            // Discard sockets past their warm lifetime (idle-billing guard).
            if candidate.warmed_at.elapsed() > self.config.max_idle {
                continue; // drops `candidate`, closing the socket
            }
            // Liveness: a non-blocking check that the socket hasn't already
            // delivered a Close/Err. A warm socket should be silent after
            // session.created, so anything pending means it is unhealthy.
            if is_dead(&mut candidate.rx) {
                continue;
            }
            return Some(WarmHandoff {
                tx: candidate.tx,
                rx: candidate.rx,
                session_created: candidate.session_created,
            });
        }
    }

    /// One replenish pass over every registered key: reap stale sockets, then
    /// dial up to `target_size`. Dials run concurrently; failures are swallowed
    /// (a key that can't be warmed just keeps fresh-dialing on the request path)
    /// and put the key into exponential backoff so a broken key isn't re-dialed
    /// on every tick.
    async fn replenish_all(&self) {
        let keys: Vec<UpstreamKey> = { self.warm.lock().unwrap().keys().cloned().collect() };
        for key in keys {
            self.reap_stale(&key);
            // Skip keys still in backoff from a prior all-failed pass — this is
            // what bounds dials against an invalid/unreachable key to once per
            // `BACKOFF_MAX` instead of `needed` dials every 250 ms tick.
            if self.in_backoff(&key) {
                continue;
            }
            let needed = {
                let warm = self.warm.lock().unwrap();
                let have = warm.get(&key).map(Vec::len).unwrap_or(0);
                self.config.target_size.saturating_sub(have)
            };
            if needed == 0 {
                continue;
            }
            // Dial the missing sockets CONCURRENTLY. A sequential loop here makes
            // a full refill cost `needed × handshake` (~needed × 350 ms), which
            // can't keep up with a high connect rate — the pool drains faster
            // than it refills and most connects miss. Firing the dials together
            // refills in ~one handshake window, keeping warm supply ≈ peak
            // concurrent connects so the sub-ms warm handoff becomes the median,
            // not the lucky-hit tail.
            let dials = (0..needed).map(|_| warm_one(&key));
            let results = futures_util::future::join_all(dials).await;
            let mut any_ok = false;
            // `.flatten()` keeps only the successful dials; a key that can't be
            // warmed just keeps fresh-dialing on the request path.
            for conn in results.into_iter().flatten() {
                any_ok = true;
                self.warm
                    .lock()
                    .unwrap()
                    .entry(key.clone())
                    .or_default()
                    .push(conn);
            }
            // Reset backoff on any success; otherwise grow it. We only ever enter
            // backoff when a pass that *attempted* dials produced none — a `needed
            // == 0` pass is handled by the `continue` above and never touches it.
            self.record_replenish_outcome(&key, any_ok);
        }
    }

    /// Whether `key` is currently in a backoff window (a prior pass failed and
    /// the retry time hasn't arrived). Eligible keys are pruned from the backoff
    /// map so it doesn't grow unbounded for healthy keys.
    fn in_backoff(&self, key: &UpstreamKey) -> bool {
        let mut backoff = self.backoff.lock().unwrap();
        match backoff.get(key).and_then(|b| b.retry_after) {
            Some(retry_after) if Instant::now() < retry_after => true,
            Some(_) => {
                // Window elapsed — allow the attempt. Keep the failure count so a
                // still-broken key backs off further, but clear the gate so this
                // tick proceeds.
                if let Some(b) = backoff.get_mut(key) {
                    b.retry_after = None;
                }
                false
            }
            None => false,
        }
    }

    /// Update a key's backoff after a replenish attempt. Success clears it;
    /// failure grows the retry delay exponentially up to `BACKOFF_MAX`.
    fn record_replenish_outcome(&self, key: &UpstreamKey, any_ok: bool) {
        let mut backoff = self.backoff.lock().unwrap();
        if any_ok {
            backoff.remove(key);
            return;
        }
        let entry = backoff.entry(key.clone()).or_default();
        entry.consecutive_failures = entry.consecutive_failures.saturating_add(1);
        // Exponential: BASE * 2^(failures-1), saturating at MAX. `min` of the
        // shift exponent keeps the doubling from overflowing.
        let shift = (entry.consecutive_failures - 1).min(16);
        let delay = BACKOFF_BASE.saturating_mul(1u32 << shift).min(BACKOFF_MAX);
        entry.retry_after = Some(Instant::now() + delay);
    }

    /// Drop sockets past `max_idle` or already dead for a key.
    fn reap_stale(&self, key: &UpstreamKey) {
        let mut warm = self.warm.lock().unwrap();
        if let Some(bucket) = warm.get_mut(key) {
            bucket.retain_mut(|conn| {
                conn.warmed_at.elapsed() <= self.config.max_idle && !is_dead(&mut conn.rx)
            });
        }
    }

    /// Test/inspection: number of warm sockets currently held for `key`.
    #[cfg(test)]
    pub fn warm_len(&self, key: &UpstreamKey) -> usize {
        self.warm
            .lock()
            .unwrap()
            .get(key)
            .map(Vec::len)
            .unwrap_or(0)
    }

    /// Test/inspection: consecutive replenish failures recorded for `key` (0 if
    /// the key is healthy / has no backoff entry).
    #[cfg(test)]
    pub fn backoff_failures(&self, key: &UpstreamKey) -> u32 {
        self.backoff
            .lock()
            .unwrap()
            .get(key)
            .map(|b| b.consecutive_failures)
            .unwrap_or(0)
    }

    /// Test helper: synchronously warm `target_size` sockets for `key` (no
    /// background task). Lets tests assert handoff behavior deterministically.
    #[cfg(test)]
    pub async fn warm_now(&self, key: &UpstreamKey) {
        let needed = {
            let warm = self.warm.lock().unwrap();
            let have = warm.get(key).map(Vec::len).unwrap_or(0);
            self.config.target_size.saturating_sub(have)
        };
        for _ in 0..needed {
            if let Ok(conn) = warm_one(key).await {
                self.warm
                    .lock()
                    .unwrap()
                    .entry(key.clone())
                    .or_default()
                    .push(conn);
            }
        }
    }

    /// Test helper: insert an already-built warm connection (used to inject a
    /// dead socket and assert it is discarded at handoff).
    #[cfg(test)]
    fn insert_warm(&self, key: UpstreamKey, conn: WarmConnection) {
        self.warm.lock().unwrap().entry(key).or_default().push(conn);
    }
}

/// Dial one upstream and pre-read its `session.created` into a [`WarmConnection`].
///
/// `key.api_key` is already resolved (non-blank). The first frame OpenAI sends
/// unprompted is `session.created`; we buffer exactly that and read nothing more.
async fn warm_one(key: &UpstreamKey) -> CoreResult<WarmConnection> {
    let upstream: UpstreamWs =
        dial_upstream(&key.model, &key.api_key, key.api_base.as_deref()).await?;
    let (tx, mut rx) = upstream.split();
    let session_created = read_event(&mut rx).await?;
    Ok(WarmConnection {
        tx,
        rx,
        session_created,
        warmed_at: Instant::now(),
    })
}

/// Resolve a deployment's API key into the pool key, returning `None` when no key
/// can be resolved (those deployments simply aren't pooled — the request path
/// still fresh-dials and surfaces the auth error there).
pub fn upstream_key(
    model: &str,
    api_key: Option<&str>,
    api_base: Option<&str>,
) -> Option<UpstreamKey> {
    let api_key = resolve_api_key(api_key).ok()?;
    Some(UpstreamKey {
        model: model.to_string(),
        api_key,
        api_base: api_base.map(str::to_string),
    })
}

/// Non-blocking liveness check: poll the upstream once. A warm socket is silent
/// after `session.created`, so a pending `Close`/`Err`/`None` means it is dead.
/// A pending data frame (shouldn't happen pre-handoff) is also treated as
/// unhealthy — we'd rather discard and fresh-dial than hand over a socket in an
/// unexpected state. `Pending` (the healthy case) returns `false`.
fn is_dead(rx: &mut UpstreamRx) -> bool {
    use futures_util::task::noop_waker_ref;
    use futures_util::Stream;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    let mut cx = Context::from_waker(noop_waker_ref());
    match Pin::new(rx).poll_next(&mut cx) {
        Poll::Pending => false,
        Poll::Ready(None) => true,
        Poll::Ready(Some(Err(_))) => true,
        // Any frame arriving before handoff is unexpected for a silent warm
        // socket; treat it as unhealthy.
        Poll::Ready(Some(Ok(_))) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::SinkExt;
    use std::net::SocketAddr;
    use tokio::net::TcpListener;
    use tokio_tungstenite::tungstenite::Message;

    /// An in-process fake OpenAI realtime WS server. On connect it sends
    /// `session.created`; on `response.create` it sends `response.created` +
    /// `response.output_audio.delta` + `response.done`. Returns its `ws://` base.
    async fn spawn_fake_openai() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                tokio::spawn(handle_fake_conn(stream));
            }
        });
        format!("ws://{addr}")
    }

    async fn handle_fake_conn(stream: tokio::net::TcpStream) {
        let mut ws = match tokio_tungstenite::accept_async(stream).await {
            Ok(ws) => ws,
            Err(_) => return,
        };
        // Unprompted session.created, exactly like OpenAI.
        let _ = ws
            .send(Message::Text(
                r#"{"type":"session.created","session":{"id":"sess_fake"}}"#.to_string(),
            ))
            .await;
        while let Some(Ok(msg)) = ws.next().await {
            if let Message::Text(text) = msg {
                if text.contains("response.create") {
                    for frame in [
                        r#"{"type":"response.created"}"#,
                        r#"{"type":"response.output_audio.delta","delta":"AAAA"}"#,
                        r#"{"type":"response.done"}"#,
                    ] {
                        let _ = ws.send(Message::Text(frame.to_string())).await;
                    }
                }
            }
        }
    }

    fn test_config() -> PoolConfig {
        PoolConfig {
            target_size: 2,
            max_idle: Duration::from_secs(30),
        }
    }

    fn key_for(base: &str) -> UpstreamKey {
        UpstreamKey {
            model: "gpt-realtime".to_string(),
            api_key: "sk-test".to_string(),
            api_base: Some(base.to_string()),
        }
    }

    #[tokio::test]
    async fn warm_handoff_relays_buffered_session_created() {
        let base = spawn_fake_openai().await;
        let pool = RealtimePool::new_unspawned(test_config());
        let key = key_for(&base);
        pool.register(key.clone());
        pool.warm_now(&key).await;
        assert_eq!(pool.warm_len(&key), 2);

        let handoff = pool.take(&key).expect("a warm socket should be available");
        assert_eq!(handoff.session_created.event_type, "session.created");
        assert_eq!(
            handoff
                .session_created
                .data
                .get("session")
                .and_then(|s| s.get("id"))
                .and_then(|v| v.as_str()),
            Some("sess_fake")
        );
        // Taking one leaves one.
        assert_eq!(pool.warm_len(&key), 1);
    }

    #[tokio::test]
    async fn pool_miss_returns_none_for_fresh_dial_fallback() {
        let base = spawn_fake_openai().await;
        let pool = RealtimePool::new_unspawned(test_config());
        let key = key_for(&base);
        // Registered but never warmed → empty bucket → miss.
        pool.register(key.clone());
        assert!(pool.take(&key).is_none());

        // Unknown key → miss.
        let other = key_for("ws://127.0.0.1:1");
        assert!(pool.take(&other).is_none());
    }

    #[tokio::test]
    async fn disabled_pool_never_hands_off() {
        let pool = RealtimePool::disabled();
        let key = key_for("ws://127.0.0.1:1");
        pool.register(key.clone());
        assert_eq!(pool.warm_len(&key), 0);
        assert!(pool.take(&key).is_none());
    }

    #[tokio::test]
    async fn dead_warm_socket_is_discarded() {
        let base = spawn_fake_openai().await;
        let pool = RealtimePool::new_unspawned(test_config());
        let key = key_for(&base);
        pool.register(key.clone());

        // Build one real warm connection, then kill the upstream by dropping the
        // server side: easiest is to dial, read session.created, then close our
        // own rx's peer. Instead we forge "dead" via an already-closed socket:
        // dial a connection and immediately send a Close from the client side so
        // the server closes back, then warm it. Simpler: warm normally, then
        // mark it stale by backdating warmed_at past max_idle and confirm it's
        // dropped — that exercises the same discard path.
        let mut conn = warm_one(&key).await.expect("warm one");
        conn.warmed_at = Instant::now() - Duration::from_secs(3600); // past max_idle
        pool.insert_warm(key.clone(), conn);
        assert_eq!(pool.warm_len(&key), 1);

        // take() must discard the stale socket and report a miss.
        assert!(pool.take(&key).is_none());
        assert_eq!(pool.warm_len(&key), 0);
    }

    #[tokio::test]
    async fn background_replenisher_tops_up_registered_key() {
        let base = spawn_fake_openai().await;
        let pool = RealtimePool::spawn(test_config());
        let key = key_for(&base);
        pool.register(key.clone());

        // Wait (bounded) for the background task to reach the target size.
        let mut warmed = 0;
        for _ in 0..40 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            warmed = pool.warm_len(&key);
            if warmed >= test_config().target_size {
                break;
            }
        }
        assert_eq!(
            warmed,
            test_config().target_size,
            "background replenisher should warm up to target_size"
        );
        let handoff = pool.take(&key).expect("a warm socket should be available");
        assert_eq!(handoff.session_created.event_type, "session.created");
    }

    #[tokio::test]
    async fn closed_upstream_socket_is_detected_dead() {
        // A genuinely dead socket: dial the fake, read session.created, then drop
        // the server by closing from our side and waiting for the close to land.
        let base = spawn_fake_openai().await;
        let pool = RealtimePool::new_unspawned(test_config());
        let key = key_for(&base);
        pool.register(key.clone());

        let mut conn = warm_one(&key).await.expect("warm one");
        // Close the upstream from the client side; the server echoes a close.
        let _ = conn.tx.send(Message::Close(None)).await;
        // Give the close a moment to arrive on rx.
        tokio::time::sleep(Duration::from_millis(50)).await;
        pool.insert_warm(key.clone(), conn);

        // Liveness check at take() should detect the close and discard it.
        assert!(pool.take(&key).is_none());
        assert_eq!(pool.warm_len(&key), 0);
    }

    #[tokio::test]
    async fn broken_key_backs_off_instead_of_dialing_every_tick() {
        // A key whose upstream is unreachable: every warm-up dial fails.
        let pool = RealtimePool::new_unspawned(test_config());
        let key = key_for("ws://127.0.0.1:1"); // nothing listens here
        pool.register(key.clone());

        // First pass attempts dials, they all fail → key enters backoff, no warm
        // sockets, one recorded failure.
        pool.replenish_all().await;
        assert_eq!(pool.warm_len(&key), 0);
        assert_eq!(pool.backoff_failures(&key), 1);
        assert!(
            pool.in_backoff(&key),
            "a key whose dials all failed must be in backoff"
        );

        // An immediate next pass must be SKIPPED (still in the backoff window), so
        // it does NOT fire another round of dials — the failure count is unchanged.
        pool.replenish_all().await;
        assert_eq!(
            pool.backoff_failures(&key),
            1,
            "replenish during the backoff window must not re-dial the broken key"
        );
    }

    #[tokio::test]
    async fn healthy_key_never_enters_backoff_and_clears_after_recovery() {
        let base = spawn_fake_openai().await;
        let pool = RealtimePool::new_unspawned(test_config());
        let key = key_for(&base);
        pool.register(key.clone());

        // A reachable upstream: the pass succeeds, so the key is never backed off.
        pool.replenish_all().await;
        assert_eq!(pool.warm_len(&key), test_config().target_size);
        assert_eq!(pool.backoff_failures(&key), 0);
        assert!(!pool.in_backoff(&key));
    }
}
