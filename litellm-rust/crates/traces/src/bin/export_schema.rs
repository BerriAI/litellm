fn main() {
    let schemas = if std::env::args().nth(1).as_deref() == Some("--requests") {
        litellm_traces::schema::request_schemas()
    } else {
        litellm_traces::schema::schemas()
    };
    println!("{}", serde_json::to_string_pretty(&schemas).unwrap());
}
