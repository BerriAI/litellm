use litellm_lens::{config::http_client, control::Control, wire, worker::Worker};

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let address = std::env::var("LITELLM_URL")?.parse()?;
    let token = std::env::var("LENS_WORKER_TOKEN")?;
    let release = std::env::var("LITELLM_RELEASE_TAG")?;
    let worker = Worker::new(Control::new(http_client()?, address, token), release);
    if !worker.run_once().await? {
        return Err(format!(
            "No compatible work was offered for protocol {}",
            wire::PROTOCOL_VERSION
        )
        .into());
    }
    Ok(())
}
