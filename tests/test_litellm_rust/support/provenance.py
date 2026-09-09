def has_rust_response_marker(response: object) -> bool:
    from litellm.rust_bridge.provenance import has_rust_response_marker as implementation

    return implementation(response)
