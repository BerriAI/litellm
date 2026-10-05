fn main() {
    let arguments = std::env::args().collect::<Vec<_>>();
    let schemas = if arguments.get(1).is_some_and(|arg| arg == "--api") {
        litellm_traces::schema::api_schemas()
    } else {
        litellm_traces::schema::schemas()
    };
    println!("{}", serde_json::to_string_pretty(&schemas).unwrap());
}
