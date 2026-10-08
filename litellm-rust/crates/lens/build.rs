fn main() {
    println!("cargo:rerun-if-changed=contract.json");
    let document: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string("contract.json").expect("Lens contract exists"),
    )
    .expect("valid JSON");
    let version = document["x-lens-protocol-version"]
        .as_u64()
        .expect("contract includes protocol version");
    let schema = serde_json::from_value(document).expect("Lens contract is valid JSON Schema");
    let mut types = typify::TypeSpace::default();
    types
        .add_root_schema(schema)
        .expect("Lens contract generates Rust types");
    let syntax = syn::parse2(types.to_stream()).expect("generated types are valid Rust");
    let output = std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    std::fs::write(
        output.join("wire.rs"),
        format!(
            "pub const PROTOCOL_VERSION: u64 = {version};\n{}",
            prettyplease::unparse(&syntax)
        ),
    )
    .expect("write generated types");
}
