use crate::{
    Error,
    control::{Control, JobClient},
    model, pipeline, wire,
};
use http::Method;
use serde::Deserialize;
use serde_json::{Value, json};
use std::time::Duration;

#[derive(Clone)]
pub struct Worker {
    control: Control,
    release: String,
}

#[derive(Deserialize)]
struct Identity {
    lens_id: String,
    job: JobIdentity,
}

#[derive(Deserialize)]
struct JobIdentity {
    id: String,
    attempts: u64,
}

impl Worker {
    pub fn new(control: Control, release: String) -> Self {
        Self { control, release }
    }

    pub async fn run_once(&self) -> Result<bool, Error> {
        let mut url = self.control.url("lens/worker/claim")?;
        url.query_pairs_mut()
            .append_pair("protocol_version", &wire::PROTOCOL_VERSION.to_string())
            .append_pair("worker_release", &self.release);
        let payload: Value = self
            .control
            .request(Method::POST, url, None::<&()>, Duration::from_secs(180))
            .await?;
        if payload.is_null() {
            return Ok(false);
        }
        let validator = jsonschema::validator_for(&model::schema("Claim")?)
            .map_err(|_| Error::InvalidRequest)?;
        let claim = serde_json::from_value::<wire::Claim>(payload.clone());
        if claim.is_err() || !validator.is_valid(&payload) {
            let identity: Identity = serde_json::from_value(payload)?;
            let client =
                JobClient::new(self.control.clone(), &identity.lens_id, &identity.job.id, 1)?
                    .with_attempt(identity.job.attempts);
            self.failure(&client, "The worker could not read this investigation. Update the worker to match the gateway, then retry.").await?;
            return Ok(true);
        }
        let mut claim = claim?;
        let client = JobClient::new(
            self.control.clone(),
            &claim.lens_id,
            &claim.job.id,
            claim.job.settings.concurrency.get() as usize,
        )?
        .with_attempt(u64::try_from(claim.job.attempts).map_err(|_| Error::InvalidRequest)?);
        let work = async {
            let sample: wire::Sample = client.get("sample").await?;
            claim.reviews = Some(client.get("reviews").await?);
            let result = pipeline::analyze(&claim, sample, client.clone()).await?;
            let _: Value = client.post("result", &result).await?;
            Ok::<_, Error>(())
        };
        let pulse = async {
            loop {
                tokio::time::sleep(Duration::from_secs(30)).await;
                match client.post::<Value>("heartbeat", &json!({})).await {
                    Ok(_) => {}
                    Err(Error::Request(_))
                    | Err(Error::Control {
                        status: 429 | 500..=599,
                        ..
                    }) => tracing::warn!("Lens heartbeat failed; retrying"),
                    Err(error) => return Err::<(), _>(error),
                }
            }
        };
        let outcome = tokio::select! { result = work => result, result = pulse => result };
        match outcome {
            Ok(()) | Err(Error::Control { status: 409, .. }) => {}
            Err(error) => self.failure(&client, &error.to_string()).await?,
        }
        Ok(true)
    }

    async fn failure(&self, client: &JobClient, message: &str) -> Result<(), Error> {
        let result = wire::Result {
            coverage: wire::Coverage::default(),
            findings: Vec::new(),
            assessments: Vec::new(),
            review_versions: Vec::new(),
            error: message.into(),
        };
        match client.post::<Value>("result", &result).await {
            Ok(_) | Err(Error::Control { status: 409, .. }) => Ok(()),
            Err(error) => Err(error),
        }
    }

    async fn slot(&self) {
        let mut delay = 2;
        loop {
            match self.run_once().await {
                Ok(true) => {
                    delay = 2;
                    continue;
                }
                Err(Error::Control { status: 409, .. }) => {
                    tracing::warn!(
                        "Lens worker version does not match the gateway; upgrade them together"
                    );
                    tokio::time::sleep(Duration::from_secs(60)).await;
                    continue;
                }
                Err(_) => tracing::warn!("Lens worker could not reach the gateway"),
                Ok(false) => {}
            }
            tokio::time::sleep(Duration::from_secs(delay)).await;
            delay = (delay * 2).min(15);
        }
    }

    pub async fn serve(self) {
        tokio::join!(self.slot(), self.slot(), self.slot());
    }
}
