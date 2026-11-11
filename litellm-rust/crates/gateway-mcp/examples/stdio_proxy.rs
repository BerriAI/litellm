use std::{error::Error, sync::Arc, time::Duration};

use litellm_gateway_mcp::{HttpConfig, NativeGateway, Registry, Server, ServerInfo, router};
use rmcp::{ServiceExt, transport::TokioChildProcess};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    let (program, arguments) = arguments
        .split_first()
        .ok_or("usage: stdio_proxy <program> [arguments...]")?;
    let mut command = tokio::process::Command::new(program);
    command.args(arguments).kill_on_drop(true);
    let upstream = ().serve(TokioChildProcess::new(command)?).await?;
    let server = Server {
        info: ServerInfo {
            server_id: "local".into(),
            server_name: "local".into(),
            alias: None,
        },
        peer: upstream.peer().clone(),
        allowed_tools: None,
    };
    let gateway = Arc::new(NativeGateway::new(
        Arc::new(Registry::new(vec![server])?),
        Duration::from_secs(30),
    ));
    let config = HttpConfig::default();
    let shutdown = config.cancellation_token.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:4000").await?;
    eprintln!("Local MCP example listening on http://127.0.0.1:4000/mcp");
    axum::serve(listener, router(gateway, config))
        .with_graceful_shutdown(async move {
            let _ = tokio::signal::ctrl_c().await;
            shutdown.cancel();
        })
        .await?;
    upstream.cancel().await?;
    Ok(())
}
