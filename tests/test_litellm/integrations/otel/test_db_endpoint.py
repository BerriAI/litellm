"""Tests for litellm/integrations/otel/model/db_endpoint.py

Prisma talks to PostgreSQL through a loopback query engine, so a DB span with no
``server.address`` gets attributed to ``localhost`` by the backend. These cover
the endpoint derivation that names the real server, for the local engine and for
remote and read-replica deployments, and pin the rule that no credential is ever
exported.
"""

from unittest.mock import patch

import pytest

from litellm.integrations.otel.model.db_endpoint import (
    DatabaseEndpoint,
    db_span_attributes,
    parse_database_endpoint,
    postgres_endpoint,
)

LOCAL_DSN = "postgresql://llmproxy:dbpassword9090@localhost:5432/litellm"
REMOTE_DSN = "postgresql://llmproxy:s3cr3t@litellm-prod.abc123.us-east-1.rds.amazonaws.com:6432/litellm?schema=reporting&sslmode=require"
REPLICA_DSN = "postgresql://reader:r3ad0nly@litellm-prod-ro.abc123.us-east-1.rds.amazonaws.com/litellm_replica"


@pytest.fixture(autouse=True)
def _clear_endpoint_cache():
    postgres_endpoint.cache_clear()
    yield
    postgres_endpoint.cache_clear()


def _resolve(service, call_type=None, database_url=None, read_replica_url=None):
    """Resolve attributes with the two DB env vars stubbed at their read site."""

    def _secret(name, default_value=None):
        return {"DATABASE_URL": database_url, "DATABASE_URL_READ_REPLICA": read_replica_url}.get(name)

    with patch("litellm.integrations.otel.model.db_endpoint.get_secret_str", side_effect=_secret):
        return dict(db_span_attributes(service, call_type))


def test_local_prisma_engine_endpoint_is_the_postgres_server_not_the_engine():
    assert parse_database_endpoint(LOCAL_DSN) == DatabaseEndpoint(
        address="localhost", port=5432, namespace="litellm"
    )


def test_remote_endpoint_keeps_host_port_and_schema_qualified_namespace():
    assert parse_database_endpoint(REMOTE_DSN) == DatabaseEndpoint(
        address="litellm-prod.abc123.us-east-1.rds.amazonaws.com",
        port=6432,
        namespace="litellm|reporting",
    )


def test_read_replica_dsn_parses_to_the_replica_host_and_database():
    assert parse_database_endpoint(REPLICA_DSN) == DatabaseEndpoint(
        address="litellm-prod-ro.abc123.us-east-1.rds.amazonaws.com",
        port=5432,
        namespace="litellm_replica",
    )


def test_default_schema_is_not_spelled_out_in_the_namespace():
    """``?schema=public`` and no schema at all are the same deployment, so they
    must not split a group-by on db.namespace."""
    assert parse_database_endpoint("postgresql://u:p@db.internal/litellm?schema=public") == parse_database_endpoint(
        "postgresql://u:p@db.internal/litellm"
    )


def test_unix_socket_host_parameter_wins_over_the_netloc():
    """libpq and the Cloud SQL connector both put the real target in ``host=``
    behind a localhost netloc, which is the attribution this module removes."""
    assert parse_database_endpoint(
        "postgresql://u:p@localhost:5432/litellm?host=/cloudsql/proj:us-east1:inst"
    ) == DatabaseEndpoint(address="/cloudsql/proj:us-east1:inst", port=5432, namespace="litellm")


def test_socket_only_dsn_without_a_netloc_host_still_resolves():
    assert parse_database_endpoint("postgresql:///litellm?host=/var/run/postgresql") == DatabaseEndpoint(
        address="/var/run/postgresql", port=5432, namespace="litellm"
    )


def test_percent_encoded_database_name_is_decoded():
    endpoint = parse_database_endpoint("postgresql://u:p@db.internal/litellm%20prod")
    assert endpoint is not None and endpoint.namespace == "litellm prod"


MISPARSED_AUTHORITY_DSNS = (
    ("postgresql://litellm:/kJ8xQz+9wT@db.internal:5432/litellm", "kJ8xQz+9wT"),
    ("postgresql://litellm:12345/aBcD@db.internal:5432/litellm", "aBcD"),
)


@pytest.mark.parametrize(("dsn", "secret"), MISPARSED_AUTHORITY_DSNS)
def test_unencoded_slash_in_password_never_yields_an_endpoint(dsn, secret):
    """An unencoded '/' truncates the authority, so urlparse reports the username
    as the host and the password tail as the database. Postgres drivers reject
    such a DSN outright, so the only safe reading is no endpoint at all."""
    assert parse_database_endpoint(dsn) is None


@pytest.mark.parametrize(("dsn", "secret"), MISPARSED_AUTHORITY_DSNS)
def test_unencoded_slash_in_password_never_reaches_a_span(dsn, secret):
    attrs = _resolve("postgres", "get_data", database_url=dsn)
    exported = " ".join(str(value) for value in attrs.values())
    assert secret not in exported
    assert "db.namespace" not in attrs
    assert "server.address" not in attrs


def test_extra_path_segment_yields_no_endpoint():
    """A database name cannot hold an unencoded '/', so a second path segment
    means the authority was mis-split even when no '@' survived into the path."""
    assert parse_database_endpoint("postgresql://db.internal:5432/litellm/extra") is None


def test_percent_encoded_password_still_resolves_the_endpoint():
    """The encoded spelling is the one a driver accepts, so it must keep working."""
    assert parse_database_endpoint("postgresql://litellm:pa%2Fssw0rd@db.internal:5432/litellm") == DatabaseEndpoint(
        address="db.internal", port=5432, namespace="litellm"
    )


def test_hostless_socket_dsn_still_names_the_database():
    """``postgresql:///litellm`` is a valid local-socket DSN that Prisma accepts,
    so the database is knowable even though no server address is."""
    assert parse_database_endpoint("postgresql:///litellm") == DatabaseEndpoint(
        address=None, port=None, namespace="litellm"
    )


def test_hostless_socket_dsn_emits_namespace_without_a_server():
    attrs = _resolve("postgres", "get_data", database_url="postgresql:///litellm")
    assert attrs["db.namespace"] == "litellm"
    assert "server.address" not in attrs
    assert "server.port" not in attrs


def test_dsn_with_neither_host_nor_database_yields_no_endpoint():
    assert parse_database_endpoint("postgresql://") is None


def test_prisma_default_schema_is_left_implicit():
    endpoint = parse_database_endpoint("postgresql://u:p@db.internal/litellm?schema=public")
    assert endpoint is not None and endpoint.namespace == "litellm"


@pytest.mark.parametrize("spelling", ["PUBLIC", "Public", "reporting"])
def test_a_non_default_schema_stays_in_the_namespace(spelling):
    """Prisma quotes the schema name, so ``?schema=PUBLIC`` provisions a second
    schema alongside ``public`` with its own tables. Case-folding them into one
    namespace would report two different schemas as the same database."""
    endpoint = parse_database_endpoint(f"postgresql://u:p@db.internal/litellm?schema={spelling}")
    assert endpoint is not None and endpoint.namespace == f"litellm|{spelling}"


def test_postgres_scheme_alias_is_accepted():
    assert parse_database_endpoint("postgres://u:p@db.internal/litellm") == DatabaseEndpoint(
        address="db.internal", port=5432, namespace="litellm"
    )


@pytest.mark.parametrize(
    "dsn",
    [
        None,
        "",
        "mysql://u:p@db.internal:3306/litellm",
        "postgresql://u:p@db.internal:not-a-port/litellm",
        "not a url at all",
    ],
)
def test_unusable_dsn_degrades_to_no_endpoint(dsn):
    assert parse_database_endpoint(dsn) is None


def test_database_without_name_or_schema_has_no_namespace():
    assert parse_database_endpoint("postgresql://u:p@db.internal:5432/") == DatabaseEndpoint(
        address="db.internal", port=5432, namespace=None
    )


def test_postgres_service_span_carries_system_operation_and_endpoint():
    assert _resolve("postgres", "get_data", database_url=REMOTE_DSN) == {
        "db.system.name": "postgresql",
        "db.system": "postgresql",
        "db.operation.name": "get_data",
        "server.address": "litellm-prod.abc123.us-east-1.rds.amazonaws.com",
        "server.port": 6432,
        "db.namespace": "litellm|reporting",
    }


def test_legacy_db_system_is_dual_emitted_for_datadog():
    """Datadog's OTLP intake infers the database span type from ``db.system``,
    not from the semconv-current ``db.system.name``."""
    assert _resolve("postgres", "get_data", database_url=LOCAL_DSN)["db.system"] == "postgresql"
    assert _resolve("redis", "set")["db.system"] == "redis"


def test_batch_write_service_is_also_attributed_to_postgres():
    attrs = _resolve("batch_write_to_db", "_PROXY_track_cost_callback", database_url=REMOTE_DSN)
    assert attrs["db.system.name"] == "postgresql"
    assert attrs["server.address"] == "litellm-prod.abc123.us-east-1.rds.amazonaws.com"


def test_redis_service_never_borrows_the_postgres_endpoint():
    assert _resolve("redis", "set", database_url=REMOTE_DSN) == {
        "db.system.name": "redis",
        "db.system": "redis",
        "db.operation.name": "set",
    }


def test_non_datastore_service_gets_no_db_attributes():
    assert _resolve("reset_budget_job", "reset_budget", database_url=REMOTE_DSN) == {}


def test_configured_read_replica_suppresses_the_endpoint_rather_than_naming_the_primary():
    """Reads are routed to the replica per Prisma call, underneath the span, so
    naming the writer would pin replica latency onto the primary."""
    attrs = _resolve("postgres", "get_data", database_url=REMOTE_DSN, read_replica_url=REPLICA_DSN)
    assert attrs == {
        "db.system.name": "postgresql",
        "db.system": "postgresql",
        "db.operation.name": "get_data",
    }


def test_endpoint_attributes_are_omitted_when_database_url_is_unset():
    assert _resolve("postgres", "get_data") == {
        "db.system.name": "postgresql",
        "db.system": "postgresql",
        "db.operation.name": "get_data",
    }


def test_blank_call_type_does_not_emit_an_empty_operation_attribute():
    assert "db.operation.name" not in _resolve("postgres", "")
    assert "db.operation.name" not in _resolve("postgres", None)


@pytest.mark.parametrize(
    ("dsn", "secrets"),
    [
        (LOCAL_DSN, ("dbpassword9090", "llmproxy")),
        (REMOTE_DSN, ("s3cr3t", "llmproxy", "sslmode")),
        (REPLICA_DSN, ("r3ad0nly", "reader")),
    ],
)
def test_no_credential_reaches_any_exported_attribute(dsn, secrets):
    attrs = _resolve("postgres", "get_data", database_url=dsn)
    assert attrs["server.address"]
    exported = " ".join(str(value) for value in attrs.values())
    for secret in secrets:
        assert secret not in exported


def test_database_url_is_resolved_once_not_per_span():
    """``get_secret_str`` can reach a configured secret manager, so resolving it
    per DB span would put a blocking lookup on the request path."""
    with patch(
        "litellm.integrations.otel.model.db_endpoint.get_secret_str",
        side_effect=lambda name, default_value=None: LOCAL_DSN if name == "DATABASE_URL" else None,
    ) as resolver:
        for _ in range(5):
            attrs = db_span_attributes("postgres", "get_data")
    assert attrs["server.address"] == "localhost"
    assert resolver.call_count == 2
