pub fn registry_key(model: &str) -> &str {
    let name = model.strip_prefix("databricks/").unwrap_or(model);
    [
        ("dbrx-instruct", "databricks-dbrx-instruct"),
        (
            "meta-llama-3.1-70b-instruct",
            "databricks-meta-llama-3-1-70b-instruct",
        ),
        (
            "meta-llama-3.1-405b-instruct",
            "databricks-meta-llama-3-1-405b-instruct",
        ),
        (
            "mixtral-8x7b-instruct-v0.1",
            "databricks-mixtral-8x7b-instruct",
        ),
        ("bge-large-en", "databricks-bge-large-en"),
        ("gte-large-en", "databricks-gte-large-en"),
        ("llama-2-70b-chat", "databricks-llama-2-70b-chat"),
    ]
    .into_iter()
    .find_map(|(prefix, key)| name.starts_with(prefix).then_some(key))
    .unwrap_or(name)
}
