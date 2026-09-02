use std::error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let protocol_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../proto");
    let protocol = protocol_root.join("litellm/python_extension/v1/extension_host.proto");
    let mut prost_config = prost_build::Config::new();
    prost_config.protoc_executable(protoc_bin_vendored::protoc_bin_path()?);
    tonic_prost_build::configure()
        .build_transport(false)
        .compile_with_config(
            prost_config,
            &[protocol.as_path()],
            &[protocol_root.as_path()],
        )?;
    println!("cargo:rerun-if-changed={}", protocol.display());
    Ok(())
}
