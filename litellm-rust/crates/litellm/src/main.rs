#[tokio::main]
async fn main() {
    litellm_ai_gateway::server::run().await;
}
