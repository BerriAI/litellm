use std::{io, time::Duration};

use crate::Error;

pub async fn within<T>(
    socket_timeout: Option<Duration>,
    operation: impl Future<Output = Result<T, Error>>,
) -> Result<T, Error> {
    let Some(limit) = socket_timeout else {
        return operation.await;
    };
    tokio::time::timeout(limit, operation)
        .await
        .unwrap_or_else(|_| {
            Err(Error::Query(sqlx::Error::Io(
                io::ErrorKind::TimedOut.into(),
            )))
        })
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::within;
    use crate::Error;

    #[tokio::test(start_paused = true)]
    async fn a_slow_operation_fails_like_a_socket_timeout() {
        let result = within(Some(Duration::from_secs(5)), async {
            tokio::time::sleep(Duration::from_secs(6)).await;
            Ok::<_, Error>(())
        })
        .await;

        assert!(
            matches!(result, Err(Error::Query(sqlx::Error::Io(error))) if error.kind() == std::io::ErrorKind::TimedOut)
        );
    }

    #[tokio::test(start_paused = true)]
    async fn no_timeout_waits_for_the_operation() {
        let result = within(None, async {
            tokio::time::sleep(Duration::from_secs(3600)).await;
            Ok::<_, Error>(7)
        })
        .await;

        assert_eq!(result.ok(), Some(7));
    }
}
