import os
import re

# Mirror of test_env_keys.py for stage e2e:
#   code keys  = os.environ/KEY refs under tests/e2e
#   source of truth = ExternalSecret litellm-provider-keys in litellm-ops
#                     (checked out to ./litellm-ops in CI, same idea as docs/)
#   fail if code has keys not on that ExternalSecret

e2e_root = "./tests/e2e"
secrets_path = (
    "./litellm-ops/apps/overlays/berrie-litellm-stage/litellm-secrets.yaml"
)

os_environ_ref_pattern = re.compile(r"""['"]os\.environ/([A-Z][A-Z0-9_]*)['"]""")
env_ref_pattern = re.compile(r"""_env_ref\(\s*['"]([A-Z][A-Z0-9_]*)['"]""")

EXCLUDED_KEYS = {
    "LITELLM_MASTER_KEY",
    "DATABASE_URL",
    "LITELLM_PROXY_URL",
    "LITELLM_CONTROL_PLANE_URL",
    "E2E_UI_USERNAME",
    "E2E_UI_PASSWORD",
    "E2E_POLL_TIMEOUT",
    "E2E_POLL_INTERVAL",
    "E2E_REQUEST_TIMEOUT",
}

SKIP_DIRS = {
    "__pycache__",
    ".pytest_cache",
    "fixtures",
    "litellm-regression-tests",
    "node_modules",
}

e2e_keys = set()

for root, dirs, files in os.walk(e2e_root):
    dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
    for file in files:
        if not file.endswith(".py"):
            continue
        path = os.path.join(root, file)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        e2e_keys.update(os_environ_ref_pattern.findall(content))
        e2e_keys.update(env_ref_pattern.findall(content))

e2e_keys -= EXCLUDED_KEYS

print("Keys expected on stage gateway (found in tests/e2e):")
for key in sorted(e2e_keys):
    print(key)

if not os.path.isfile(secrets_path):
    raise Exception(
        f"Error reading litellm-ops secrets: missing {secrets_path}, "
        f"cwd files - {os.listdir('.')}"
    )

with open(secrets_path, encoding="utf-8") as f:
    secrets_content = f.read()

provider_block = re.search(
    r"(?ms)^kind: ExternalSecret\nmetadata:\n  name: litellm-provider-keys\n.*?^(?:---|\Z)",
    secrets_content,
)
if provider_block is None:
    raise Exception(
        f"ExternalSecret litellm-provider-keys not found in {secrets_path}"
    )

mounted_keys = set(
    re.findall(
        r"(?m)^\s+- secretKey: ([A-Z][A-Z0-9_]+)\s*$",
        provider_block.group(0),
    )
)

print(f"\nmounted_keys: {mounted_keys}")

missing = e2e_keys - mounted_keys

if missing:
    raise Exception(
        f"\nKeys not on stage ExternalSecret litellm-provider-keys: {missing}"
    )
else:
    print(
        "\nAll e2e os.environ keys are on litellm-provider-keys. - {}".format(
            e2e_keys
        )
    )
