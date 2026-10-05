fn main() {
    let document =
        litellm_traces::api::openapi::document(litellm_traces_clickhouse::wire_schema::schemas());
    println!("{}", document.to_pretty_json().unwrap());
}
