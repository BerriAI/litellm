fn main() {
    println!(
        "{}",
        serde_json::to_string_pretty(&litellm_traces::schema::schemas()).unwrap()
    );
}
