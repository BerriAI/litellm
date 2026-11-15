use std::time::{Duration, Instant, SystemTime};

use crate::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Call {
    pub call_type: &'static str,
    pub table: Option<&'static str>,
    pub operation: Option<&'static str>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Succeeded,
    Failed(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryEvent {
    pub call: Call,
    pub started_at: SystemTime,
    pub duration: Duration,
    pub outcome: Outcome,
}

pub trait Observer: Send + Sync {
    fn record(&self, event: QueryEvent);
}

pub async fn observe<T>(
    observer: &dyn Observer,
    call: Call,
    query: impl Future<Output = Result<T, Error>>,
) -> Result<T, Error> {
    let started_at = SystemTime::now();
    let clock = Instant::now();
    let result = query.await;
    let outcome = match &result {
        Ok(_) => Some(Outcome::Succeeded),
        Err(error @ Error::Query(_)) => Some(Outcome::Failed(
            litellm_error::Report::new(error).to_string(),
        )),
        Err(_) => None,
    };
    if let Some(outcome) = outcome {
        observer.record(QueryEvent {
            call,
            started_at,
            duration: clock.elapsed(),
            outcome,
        });
    }
    result
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::{Call, Observer, Outcome, QueryEvent, observe};
    use crate::Error;

    const CALL: Call = Call {
        call_type: "add_key_spend",
        table: Some("LiteLLM_VerificationToken"),
        operation: Some("update"),
    };

    #[derive(Default)]
    struct Recording(Mutex<Vec<QueryEvent>>);

    impl Observer for Recording {
        fn record(&self, event: QueryEvent) {
            self.0.lock().expect("recording lock").push(event);
        }
    }

    impl Recording {
        fn outcomes(&self) -> Vec<(Call, Outcome)> {
            self.0
                .lock()
                .expect("recording lock")
                .iter()
                .map(|event| (event.call, event.outcome.clone()))
                .collect()
        }
    }

    #[tokio::test]
    async fn records_a_successful_round_trip() {
        let recording = Recording::default();

        let value = observe(&recording, CALL, async { Ok::<_, Error>(7) }).await;

        assert_eq!(value.ok(), Some(7));
        assert_eq!(recording.outcomes(), vec![(CALL, Outcome::Succeeded)]);
    }

    #[tokio::test]
    async fn records_a_database_failure() {
        let recording = Recording::default();

        let result = observe(&recording, CALL, async {
            Err::<(), _>(Error::from(sqlx::Error::PoolTimedOut))
        })
        .await;

        assert!(result.is_err());
        let outcomes = recording.outcomes();
        assert_eq!(outcomes.len(), 1);
        assert!(
            matches!(&outcomes[0], (call, Outcome::Failed(message)) if *call == CALL && message.contains("pool timed out"))
        );
    }

    #[tokio::test]
    async fn skips_failures_that_never_reached_the_database() {
        let recording = Recording::default();

        let result = observe(&recording, CALL, async {
            Err::<(), _>(Error::InvalidSpendBatch { ids: 1, costs: 2 })
        })
        .await;

        assert!(result.is_err());
        assert!(recording.outcomes().is_empty());
    }
}
