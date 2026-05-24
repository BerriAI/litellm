import yaml

from litellm_provisioning_mcp import resources


def _load(manifest: str) -> list[dict]:
    return list(yaml.safe_load_all(manifest))


def test_master_key_secret_is_valid_and_prefixed():
    manifest, name = resources.master_key_secret("rel")
    docs = _load(manifest)
    assert name == "rel-master-key"
    assert len(docs) == 1
    secret = docs[0]
    assert secret["kind"] == "Secret"
    assert secret["stringData"]["master-key"].startswith("sk-")
    assert secret["metadata"]["labels"]["litellm.ai/release"] == "rel"


def test_postgres_manifests_and_connection():
    manifest, conn = resources.postgres("rel")
    docs = _load(manifest)
    kinds = sorted(d["kind"] for d in docs)
    assert kinds == ["Deployment", "Secret", "Service"]
    assert conn.host == "rel-postgres"
    assert conn.port == 5432
    assert conn.dbname == "litellm"
    assert conn.secret_name == "rel-postgres"
    # Deployment selector is a subset of the pod labels.
    deployment = next(d for d in docs if d["kind"] == "Deployment")
    selector = deployment["spec"]["selector"]["matchLabels"]
    pod_labels = deployment["spec"]["template"]["metadata"]["labels"]
    assert selector.items() <= pod_labels.items()


def test_redis_manifests_and_connection():
    manifest, conn = resources.redis("rel")
    docs = _load(manifest)
    assert sorted(d["kind"] for d in docs) == ["Deployment", "Service"]
    assert conn.host == "rel-redis"
    assert conn.port == 6379


def test_selector_label_targets_release():
    assert resources.selector_label("rel") == "litellm.ai/release=rel"
