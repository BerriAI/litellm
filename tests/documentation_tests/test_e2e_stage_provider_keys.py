import os
import re

import yaml

# Mirror of test_env_keys.py for stage e2e:
#   code keys  = os.environ/KEY refs under tests/e2e
#   source of truth = ExternalSecret litellm-provider-keys from litellm-ops
#                     (CI drops that yaml at ./litellm-secrets.yaml)
#   fail if code has keys not on that ExternalSecret

e2e_root = "./tests/e2e"
secrets_path = "./litellm-secrets.yaml"

os_environ_ref_pattern = re.compile(r"""['"]os\.environ/([A-Z][A-Z0-9_]*)['"]""")
env_ref_pattern = re.compile(r"""_env_ref\(\s*['"]([A-Z][A-Z0-9_]*)['"]""")
secret_key_pattern = re.compile(r"^[A-Z][A-Z0-9_]*$")

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
        f"Error reading stage secrets: missing {secrets_path}, "
        f"cwd files - {os.listdir('.')}"
    )

with open(secrets_path, encoding="utf-8") as f:
    secrets_content = f.read()

provider_doc = None
for doc in yaml.safe_load_all(secrets_content):
    if not isinstance(doc, dict):
        continue
    if doc.get("kind") != "ExternalSecret":
        continue
    metadata = doc.get("metadata")
    if not isinstance(metadata, dict):
        continue
    if metadata.get("name") == "litellm-provider-keys":
        provider_doc = doc
        break

if provider_doc is None:
    raise Exception(
        f"ExternalSecret litellm-provider-keys not found in {secrets_path}"
    )

spec = provider_doc.get("spec")
if not isinstance(spec, dict):
    raise Exception(
        f"ExternalSecret litellm-provider-keys has no spec in {secrets_path}"
    )

data = spec.get("data")
if not isinstance(data, list):
    raise Exception(
        f"ExternalSecret litellm-provider-keys has no data list in {secrets_path}"
    )

mounted_keys = {
    secret_key
    for item in data
    if isinstance(item, dict)
    for secret_key in [item.get("secretKey")]
    if isinstance(secret_key, str) and secret_key_pattern.fullmatch(secret_key)
}

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
