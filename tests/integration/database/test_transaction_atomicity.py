import os
import uuid
from contextlib import ExitStack
from hashlib import sha256
from typing import Final

import psycopg
import pytest
from psycopg import sql

from integration._support.client import Gateway
from integration._support.database import read_rows


@pytest.mark.covers("other.database.access_group.failed_second_write_rolls_back_first")
def test_access_group_second_key_constraint_failure_rolls_back_all_writes(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        outside: Final = scenario.model()
        keys: Final = (scenario.key(models=[outside]), scenario.key(models=[outside]))
        tokens: Final = [sha256(key.encode()).hexdigest() for key in keys]
        name: Final = f"integration-{uuid.uuid4().hex}"
        constraint: Final = f"integration_reject_{uuid.uuid4().hex}"
        witness: Final = constraint + "_seq"
        check_function: Final = constraint + "_check"
        body: Final = {"access_group_name": name, "access_model_names": [model], "assigned_key_ids": tokens}
        def remove_partial_group() -> None:
            for row in read_rows('SELECT access_group_id FROM "LiteLLM_AccessGroupTable" WHERE access_group_name=%s', (name,)):
                response: Final = gateway.request("DELETE", f"/v1/access_group/{row['access_group_id']}")
                assert response.status_code == 204, response.text
            assert read_rows('SELECT access_group_id FROM "LiteLLM_AccessGroupTable" WHERE access_group_name=%s', (name,)) == []

        scenario.cleanups.callback(remove_partial_group)
        before: Final = read_rows('SELECT token, access_group_ids FROM "LiteLLM_VerificationToken" WHERE token=ANY(%s) ORDER BY token', (tokens,))
        with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection, ExitStack() as cleanup:
            connection.execute(sql.SQL("CREATE SEQUENCE {}").format(sql.Identifier(witness)))
            cleanup.callback(connection.execute, sql.SQL("DROP SEQUENCE {}").format(sql.Identifier(witness)))
            connection.execute(sql.SQL("CREATE FUNCTION {}(text[]) RETURNS boolean LANGUAGE plpgsql AS $$ BEGIN IF cardinality($1)>0 THEN PERFORM nextval({}); RETURN false; END IF; RETURN true; END $$").format(sql.Identifier(check_function), sql.Literal(witness)))
            cleanup.callback(connection.execute, sql.SQL("DROP FUNCTION {}(text[])").format(sql.Identifier(check_function)))
            connection.execute(sql.SQL('ALTER TABLE "LiteLLM_VerificationToken" ADD CONSTRAINT {} CHECK (token <> {} OR {}(access_group_ids))').format(sql.Identifier(constraint), sql.Literal(tokens[1]), sql.Identifier(check_function)))
            cleanup.callback(connection.execute, sql.SQL('ALTER TABLE "LiteLLM_VerificationToken" DROP CONSTRAINT {}').format(sql.Identifier(constraint)))
            try:
                assert connection.execute(sql.SQL("SELECT is_called FROM {}").format(sql.Identifier(witness))).fetchone() == (False,)
                failed: Final = gateway.request("POST", "/v1/access_group", body)
                assert failed.status_code == 500, failed.text
                assert connection.execute(sql.SQL("SELECT is_called FROM {}").format(sql.Identifier(witness))).fetchone() == (True,)
                assert read_rows('SELECT access_group_id FROM "LiteLLM_AccessGroupTable" WHERE access_group_name=%s', (name,)) == []
                assert read_rows('SELECT token, access_group_ids FROM "LiteLLM_VerificationToken" WHERE token=ANY(%s) ORDER BY token', (tokens,)) == before
                for key in keys:
                    denied: Final = gateway.request("POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "rolled back grant"}]}, key=key)
                    assert denied.status_code == 403 and denied.json()["error"]["type"] == "key_model_access_denied", denied.text
            finally:
                cleanup.close()
        created: Final = gateway.request("POST", "/v1/access_group", body)
        assert created.status_code == 201, created.text
        identity: Final = created.json()["access_group_id"]
        try:
            for key in keys:
                assert gateway.chat(model, key=key)["usage"]["total_tokens"] == 40
        finally:
            deleted: Final = gateway.request("DELETE", f"/v1/access_group/{identity}")
            assert deleted.status_code == 204, deleted.text
            assert read_rows('SELECT access_group_id FROM "LiteLLM_AccessGroupTable" WHERE access_group_id=%s', (identity,)) == []
        assert read_rows('SELECT token, access_group_ids FROM "LiteLLM_VerificationToken" WHERE token=ANY(%s) ORDER BY token', (tokens,)) == before
        assert read_rows('SELECT conname FROM pg_constraint WHERE conname=%s', (constraint,)) == []
