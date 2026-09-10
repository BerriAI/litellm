mod proof;

use proof::build_proof;

fn main() {
    let proof = build_proof();
    let json = serde_json::to_string_pretty(&proof).unwrap();
    println!("{json}");
    println!(
        "litellm-core error type: {}",
        std::any::type_name::<litellm_core::Error>()
    );
}
