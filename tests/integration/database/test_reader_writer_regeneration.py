import os
import uuid
from hashlib import sha256
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Final
from urllib.parse import urlsplit, urlunsplit

import psycopg
import pytest
from psycopg import sql

from integration._support.client import Gateway, delete_key_if_present, eventually, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy


@pytest.mark.covers("other.database.regeneration.writer_updates_dependent_grants")
def test_key_regeneration_uses_writer_with_a_real_readonly_reader(gateway: Gateway, tmp_path: Path) -> None:
    role: Final = f"integration_reader_{uuid.uuid4().hex}"
    url: Final = os.environ["DATABASE_URL"]
    parsed: Final = urlsplit(url)
    reader_url: Final = urlunsplit(
        parsed._replace(netloc=f"{role}:integration-reader-password@{parsed.hostname}:{parsed.port}")
    )
    with psycopg.connect(url, autocommit=True) as admin:
        admin.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD 'integration-reader-password' NOSUPERUSER NOINHERIT").format(
                sql.Identifier(role)
            )
        )
        try:
            admin.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(role)))
            admin.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA public TO {}").format(sql.Identifier(role)))
            admin.execute(sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(sql.Identifier(role)))
            with psycopg.connect(reader_url, autocommit=True) as reader:
                assert reader.execute("SHOW transaction_read_only").fetchone() == ("on",)
                with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
                    reader.execute('UPDATE "LiteLLM_VerificationToken" SET blocked = true WHERE false')
            with owned_proxy(gateway, tmp_path, {"DATABASE_URL_READ_REPLICA": reader_url}) as candidate:
                assert read_rows("SELECT pid FROM pg_stat_activity WHERE usename=%s", (role,)), (
                    "Candidate reader was never connected"
                )
                with gateway.scenario() as scenario:
                    model: Final = scenario.model()
                    outside: Final = scenario.model()
                    old: Final = string_value(candidate.post("/key/generate", {"models": [outside]})["key"])
                    new: Final = f"sk-integration-{uuid.uuid4().hex}"
                    scenario.cleanups.callback(delete_key_if_present, gateway, old)
                    scenario.cleanups.callback(delete_key_if_present, gateway, new)
                    old_hash: Final = sha256(old.encode()).hexdigest()
                    before: Final = candidate.request(
                        "POST",
                        "/v1/chat/completions",
                        {"model": model, "messages": [{"role": "user", "content": "no grant yet"}]},
                        key=old,
                    )
                    assert before.status_code == 403 and before.json()["error"]["type"] == "key_model_access_denied", (
                        before.text
                    )
                    response: Final = candidate.request(
                        "POST",
                        "/v1/access_group",
                        {
                            "access_group_name": f"integration-{uuid.uuid4().hex}",
                            "access_model_names": [model],
                            "assigned_key_ids": [old_hash],
                        },
                    )
                    assert response.status_code == 201, response.text
                    group: Final = string_value(response.json()["access_group_id"])
                    try:
                        with psycopg.connect(url) as blocker, ThreadPoolExecutor(max_workers=1) as executor:
                            blocker.execute('LOCK TABLE "LiteLLM_AccessGroupTable" IN ACCESS EXCLUSIVE MODE')
                            pending: Final = executor.submit(candidate.request, "GET", f"/v1/access_group/{group}")
                            try:
                                reached: Final = eventually(
                                    lambda: read_rows(
                                        "SELECT usename FROM pg_stat_activity WHERE %s=ANY(pg_blocking_pids(pid)) "
                                        "AND usename=%s AND query LIKE 'SELECT%%'",
                                        (blocker.info.backend_pid, role),
                                    ),
                                    bool,
                                    seconds=3,
                                )
                                assert reached == [{"usename": role}]
                            finally:
                                blocker.rollback()
                            selected: Final = pending.result(timeout=5)
                            assert selected.status_code == 200 and selected.json()["access_group_id"] == group, (
                                selected.text
                            )
                        assert candidate.chat(model, key=old)["usage"]["total_tokens"] == 40
                        regenerated: Final = candidate.post(
                            "/key/regenerate", {"key": old, "new_key": new, "grace_period": "0s"}
                        )
                        assert regenerated["key"] == new
                        new_hash: Final = sha256(new.encode()).hexdigest()
                        assert new != old
                        assert read_rows(
                            'SELECT assigned_key_ids FROM "LiteLLM_AccessGroupTable" WHERE access_group_id=%s', (group,)
                        ) == [{"assigned_key_ids": [new_hash]}]
                        assert read_rows(
                            'SELECT token, access_group_ids FROM "LiteLLM_VerificationToken" WHERE token=ANY(%s)',
                            ([old_hash, new_hash],),
                        ) == [{"token": new_hash, "access_group_ids": [group]}]
                        assert candidate.chat(model, key=new)["usage"]["total_tokens"] == 40
                        assert candidate.chat(outside, key=new)["usage"]["total_tokens"] == 40
                        denied: Final = candidate.request(
                            "POST",
                            "/v1/chat/completions",
                            {"model": model, "messages": [{"role": "user", "content": "rotated key"}]},
                            key=old,
                        )
                        assert (
                            denied.status_code == 401 and denied.json()["error"]["type"] == "token_not_found_in_db"
                        ), denied.text
                    finally:
                        deleted: Final = gateway.request("DELETE", f"/v1/access_group/{group}")
                        assert deleted.status_code == 204, deleted.text
                        assert (
                            read_rows(
                                'SELECT access_group_id FROM "LiteLLM_AccessGroupTable" WHERE access_group_id=%s',
                                (group,),
                            )
                            == []
                        )
        finally:
            admin.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
            admin.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))
    assert read_rows("SELECT rolname FROM pg_roles WHERE rolname=%s", (role,)) == []
