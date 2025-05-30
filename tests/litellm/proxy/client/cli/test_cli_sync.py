from typing import Set
import litellm.proxy._types as proxy_types

# 1. Get all fields from UpdateKeyRequest and its bases
def get_update_key_request_fields() -> Set[str]:
    fields = set()
    model = proxy_types.UpdateKeyRequest
    for base in model.__mro__:
        if hasattr(base, "model_fields"):
            fields.update(base.model_fields.keys())
    return fields

# 2. Get all CLI options/args for the update command
def get_cli_update_args() -> Set[str]:
    import litellm.proxy.client.cli.commands.keys as keys_cli

    update_cmd = getattr(keys_cli, "update", None)
    if update_cmd is None:
        raise Exception("No 'update' command found in CLI keys.py")

    cli_args = set()
    for param in update_cmd.params:
        for opt in param.opts:
            name = opt.lstrip("-").replace("-", "_")
            cli_args.add(name)
    return cli_args

def test_cli_update_in_sync_with_api():
    api_fields = get_update_key_request_fields()
    cli_fields = get_cli_update_args()

    # These fields are required by the API but not present in the CLI
    missing_in_cli = api_fields - cli_fields
    # These fields are present in the CLI but not required by the API
    extra_in_cli = cli_fields - api_fields

    # Allow some fields to be intentionally omitted (e.g., internal fields)
    allowed_missing = {"metadata"}  # example, adjust as needed

    assert not (missing_in_cli - allowed_missing), (
        f"Fields missing in CLI update command: {missing_in_cli - allowed_missing}"
    )
    # Optionally, check for extra fields
    assert not extra_in_cli, f"Extra fields in CLI update command: {extra_in_cli}" 