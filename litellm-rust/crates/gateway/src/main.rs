use std::error::Error;

use litellm_config::Config;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let config_path = std::env::var("LITELLM_CONFIG").unwrap_or_else(|_| "config.yaml".into());
    let config = Config::load(config_path)?;
    let inference = litellm_gateway::build_inference(&config)?;
    let host = std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".into());
    let port = std::env::var("PORT")
        .unwrap_or_else(|_| "4000".into())
        .parse::<u16>()?;
    let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;

    axum::serve(listener, litellm_gateway::router(inference, &config)).await?;
    Ok(())
}
