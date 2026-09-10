pub fn dependency_report() -> String {
    format!(
        "LiteLLM Rust error type: {}",
        std::any::type_name::<litellm_core::Error>()
    )
}
