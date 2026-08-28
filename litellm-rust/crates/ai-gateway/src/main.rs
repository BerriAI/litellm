use litellm_ai_gateway::server::{ServerOptions, run};

#[tokio::main]
async fn main() -> std::process::ExitCode {
    match run(ServerOptions::from_env()).await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("server error: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
