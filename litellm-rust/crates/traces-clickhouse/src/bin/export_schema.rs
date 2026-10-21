fn main() {
    println!(
        "{}",
        serde_json::to_string_pretty(&litellm_traces_clickhouse::wire_schema::schemas()).unwrap()
    );
}
