use clap::Parser;
use litellm_ai_gateway::server::{ServerOptions, run};

#[derive(Debug, Parser)]
#[command(name = "litellm")]
struct Cli {
    #[arg(long, env = "LITELLM_CONFIG_PATH", value_name = "PATH")]
    config: Option<String>,
}

#[tokio::main]
async fn main() -> std::process::ExitCode {
    let cli: Cli = Cli::parse();
    let options: ServerOptions = ServerOptions {
        config_path: cli.config,
    };
    match run(options).await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("server error: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Cli;
    use clap::Parser;

    #[test]
    fn parses_config_path() {
        let cli: Cli = Cli::try_parse_from(["litellm", "--config", "config.yaml"])
            .expect("valid CLI arguments");

        assert_eq!(cli.config.as_deref(), Some("config.yaml"));
    }
}
