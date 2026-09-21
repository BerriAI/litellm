use std::io::Read;

use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Input {
    model_alias: String,
    provider_model: String,
    api_base: String,
    body: Value,
}

#[tokio::main]
async fn main() {
    let mut input = String::new();
    if let Err(error) = std::io::stdin().read_to_string(&mut input) {
        fail(error);
    }
    let input: Input = match serde_json::from_str(&input) {
        Ok(input) => input,
        Err(error) => fail(error),
    };
    let result = litellm_ai_gateway::trace_parity::traced_messages_request(
        input.model_alias,
        input.provider_model,
        input.api_base,
        input.body,
    )
    .await;
    match serde_json::to_string(&result) {
        Ok(result) => println!("{result}"),
        Err(error) => fail(error),
    }
}

fn fail(error: impl std::fmt::Display) -> ! {
    eprintln!("{error}");
    std::process::exit(1)
}
