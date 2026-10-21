use crate::{Error, control::JobClient, wire};
use std::sync::Arc;
use tokio::sync::Mutex;

pub struct Tracker {
    client: JobClient,
    activity: Mutex<wire::Activity>,
}

impl Tracker {
    pub async fn start(
        client: &JobClient,
        id: String,
        phase: wire::ActivityPhase,
        label: String,
        execution_ids: Vec<String>,
    ) -> Result<Arc<Self>, Error> {
        let tracker = Arc::new(Self {
            client: client.clone(),
            activity: Mutex::new(wire::Activity {
                id,
                phase,
                label,
                execution_ids,
                started_at: chrono::Utc::now(),
                operations: Vec::new(),
                tool_calls: Vec::new(),
                finished: false,
            }),
        });
        tracker.publish(&*tracker.activity.lock().await).await?;
        Ok(tracker)
    }

    async fn publish(&self, activity: &wire::Activity) -> Result<(), Error> {
        self.client
            .progress(&wire::Progress {
                activity: Some(activity.clone()),
                ..Default::default()
            })
            .await
    }

    pub async fn change(&self, operation: &str, started: bool) -> Result<(), Error> {
        let mut activity = self.activity.lock().await;
        let name: wire::ActivityOperationsItem = serde_json::from_value(operation.into())?;
        if started {
            activity.operations.push(name);
            if operation != "model" {
                let name: wire::ToolCountName = serde_json::from_value(operation.into())?;
                match activity
                    .tool_calls
                    .iter_mut()
                    .find(|count| count.name == name)
                {
                    Some(count) => count.calls += 1,
                    None => activity.tool_calls.push(wire::ToolCount { name, calls: 1 }),
                }
            }
        } else if let Some(index) = activity
            .operations
            .iter()
            .position(|current| current == &name)
        {
            activity.operations.remove(index);
        }
        self.publish(&activity).await
    }

    pub async fn finish(&self) -> Result<Vec<wire::ToolCount>, Error> {
        let mut activity = self.activity.lock().await;
        activity.finished = true;
        activity.operations.clear();
        self.publish(&activity).await?;
        Ok(activity.tool_calls.clone())
    }
}
