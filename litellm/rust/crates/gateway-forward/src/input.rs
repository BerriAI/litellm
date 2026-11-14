use std::io::{self, Read};

use tokio::sync::oneshot;

/// Resolves once `input` reaches end of file or fails to read. The Python supervisor holds the only
/// write end of the gateway's stdin, so EOF means it is gone and the gateway drains as on SIGTERM.
///
/// The read runs on a plain thread because a blocked read on tokio's blocking pool would hold up
/// runtime shutdown after a signal had already stopped the gateway.
pub fn input_closed(mut input: impl Read + Send + 'static) -> impl Future<Output = ()> + Send {
    let (closed, on_closed) = oneshot::channel::<()>();
    let reader = std::thread::Builder::new()
        .name("stdin-eof".into())
        .spawn(move || {
            let _ = io::copy(&mut input, &mut io::sink());
            let _ = closed.send(());
        });
    async move {
        match reader {
            Ok(_) => {
                let _ = on_closed.await;
            }
            Err(error) => {
                tracing::warn!(%error, "could not watch stdin, so only signals stop the gateway");
                std::future::pending::<()>().await;
            }
        }
    }
}
