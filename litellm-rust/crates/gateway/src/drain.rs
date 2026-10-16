//! Serving with a graceful drain: on the first shutdown signal readiness turns 503, then after
//! `unready_delay` the listener closes and in-flight requests and open streams get up to
//! `window` to finish. A second signal, or the window running out, ends the drain early.

use std::{fmt::Debug, future::Future, time::Duration};

use axum::{Router, serve::Listener};
use strum::IntoStaticStr;
use tokio_util::sync::CancellationToken;

use crate::health::Draining;

#[derive(Clone, Copy, Debug)]
pub struct DrainSettings {
    /// How long readiness reports 503 while new connections are still accepted, so a load
    /// balancer that polls readiness takes the instance out before its connections are refused.
    pub unready_delay: Duration,
    /// The longest in-flight work may run after the listener closes.
    pub window: Duration,
}

/// How serving ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq, IntoStaticStr)]
pub enum Outcome {
    /// The server stopped before any shutdown signal.
    #[strum(serialize = "stopped")]
    Stopped,
    /// Every in-flight request finished within the drain window.
    #[strum(serialize = "drained")]
    Drained,
    /// The drain window ran out with requests still in flight.
    #[strum(serialize = "window_elapsed")]
    WindowElapsed,
    /// A second shutdown signal arrived during the drain.
    #[strum(serialize = "interrupted")]
    Interrupted,
}

/// Serves `app` until `signal` resolves, then drains. `stop_accepting` is cancelled when the
/// listener closes, so long-lived work tied to it (MCP sessions) winds down with the drain.
/// `signal` is called again during the drain; a second signal stops waiting.
pub async fn serve<L, S, F>(
    listener: L,
    app: Router,
    draining: Draining,
    settings: DrainSettings,
    stop_accepting: CancellationToken,
    mut signal: S,
) -> std::io::Result<Outcome>
where
    L: Listener,
    L::Addr: Debug,
    S: FnMut() -> F,
    F: Future<Output = ()>,
{
    let server = axum::serve(listener, app)
        .with_graceful_shutdown(stop_accepting.clone().cancelled_owned())
        .into_future();
    tokio::pin!(server);

    tokio::select! {
        result = &mut server => return result.map(|()| Outcome::Stopped),
        () = signal() => {}
    }
    draining.start();
    tracing::info!(
        unready_delay_s = settings.unready_delay.as_secs_f64(),
        drain_window_s = settings.window.as_secs_f64(),
        "shutdown signal received, readiness now reports 503"
    );

    let second_signal = signal();
    tokio::pin!(second_signal);
    tokio::select! {
        result = &mut server => return result.map(|()| Outcome::Drained),
        () = tokio::time::sleep(settings.unready_delay) => {}
        () = &mut second_signal => {
            stop_accepting.cancel();
            return Ok(Outcome::Interrupted);
        }
    }

    stop_accepting.cancel();
    tracing::info!("stopped accepting connections, waiting for in-flight requests");
    let outcome = tokio::select! {
        result = &mut server => result.map(|()| Outcome::Drained)?,
        () = tokio::time::sleep(settings.window) => Outcome::WindowElapsed,
        () = &mut second_signal => Outcome::Interrupted,
    };
    match outcome {
        Outcome::Drained => tracing::info!("drain complete"),
        _ => tracing::warn!(
            outcome = <&'static str>::from(outcome),
            "drain ended with requests still in flight"
        ),
    }
    Ok(outcome)
}
