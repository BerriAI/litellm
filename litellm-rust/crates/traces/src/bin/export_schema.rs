fn main() {
    let schemas = match std::env::args().nth(1).as_deref() {
        Some("--requests") => litellm_traces::schema::request_schemas(),
        Some("--responses") => litellm_traces::schema::response_schemas(),
        _ => litellm_traces::schema::schemas(),
    };
    println!("{}", serde_json::to_string_pretty(&schemas).unwrap());
}
