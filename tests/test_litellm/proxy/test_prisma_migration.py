import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from litellm.proxy import prisma_migration


def _completed(returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(("prisma", "generate"), returncode, stdout="", stderr="")


class _RecordingServer:
    def __init__(self, exit_code: int | None = None) -> None:
        self.calls: list[tuple[tuple[str, ...], bool]] = []
        self.exit_code = exit_code

    def __call__(self, args: tuple[str, ...], standalone_mode: bool) -> None:
        self.calls.append((args, standalone_mode))
        if self.exit_code is not None:
            raise SystemExit(self.exit_code)


class _RecordingGenerator:
    def __init__(self, returncode: int = 0) -> None:
        self.calls = 0
        self.returncode = returncode

    def __call__(self) -> subprocess.CompletedProcess[str]:
        self.calls += 1
        return _completed(self.returncode)


class TestPrismaMigration:
    def test_main_enforces_migration_check_by_default(self) -> None:
        server = _RecordingServer()

        with patch.dict(os.environ, {}, clear=True):
            assert (
                prisma_migration.main(
                    start_server=server,
                    client_is_current=lambda: False,
                    generate_client=_RecordingGenerator(),
                )
                == 0
            )

        assert server.calls == [
            (("--skip_server_startup", "--enforce_prisma_migration_check"), False)
        ]

    def test_main_disables_migration_check_when_explicitly_false(self) -> None:
        server = _RecordingServer()

        with patch.dict(os.environ, {"ENFORCE_PRISMA_MIGRATION_CHECK": "false"}, clear=True):
            assert (
                prisma_migration.main(
                    start_server=server,
                    client_is_current=lambda: False,
                    generate_client=_RecordingGenerator(),
                )
                == 0
            )

        assert server.calls == [(("--skip_server_startup",), False)]

    @pytest.mark.parametrize("env", [{}, {"ENFORCE_PRISMA_MIGRATION_CHECK": "false"}])
    def test_main_exits_zero_when_only_prisma_generate_fails(self, env: dict[str, str]) -> None:
        generator = _RecordingGenerator(returncode=1)

        with patch.dict(os.environ, env, clear=True):
            assert (
                prisma_migration.main(
                    start_server=_RecordingServer(),
                    client_is_current=lambda: False,
                    generate_client=generator,
                )
                == 0
            )

        assert generator.calls == 1

    def test_main_propagates_migration_failure(self) -> None:
        generator = _RecordingGenerator()

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(SystemExit, match="1"):
                prisma_migration.main(
                    start_server=_RecordingServer(exit_code=1),
                    client_is_current=lambda: False,
                    generate_client=generator,
                )

        assert generator.calls == 0

    def test_main_skips_prisma_generate_when_the_client_is_already_current(self) -> None:
        server = _RecordingServer()
        generator = _RecordingGenerator()

        with patch.dict(os.environ, {}, clear=True):
            assert (
                prisma_migration.main(
                    start_server=server,
                    client_is_current=lambda: True,
                    generate_client=generator,
                )
                == 0
            )

        assert len(server.calls) == 1
        assert generator.calls == 0


class TestClientAlreadyGeneratedFrom:
    def _schema(self, tmp_path: Path, body: str) -> Path:
        schema = tmp_path / "source" / "schema.prisma"
        schema.parent.mkdir()
        schema.write_text(body)
        return schema

    def _client_dir(self, tmp_path: Path, body: str | None = None) -> Path:
        client_dir = tmp_path / "client"
        client_dir.mkdir()
        if body is not None:
            (client_dir / "schema.prisma").write_text(body)
        return client_dir

    def test_a_client_generated_from_the_same_schema_is_current(self, tmp_path: Path) -> None:
        body = "model Foo {\n  id String @id\n}\n"
        schema = self._schema(tmp_path, body)

        assert (
            prisma_migration.client_already_generated_from(
                schema, self._client_dir(tmp_path, body)
            )
            is True
        )

    def test_a_client_generated_from_a_different_schema_is_not_current(
        self, tmp_path: Path
    ) -> None:
        schema = self._schema(tmp_path, "model Foo {\n  id String @id\n}\n")

        assert (
            prisma_migration.client_already_generated_from(
                schema, self._client_dir(tmp_path, "model Foo {\n  id Int @id\n}\n")
            )
            is False
        )

    def test_an_ungenerated_client_is_not_current(self, tmp_path: Path) -> None:
        schema = self._schema(tmp_path, "model Foo {\n  id String @id\n}\n")

        assert (
            prisma_migration.client_already_generated_from(schema, self._client_dir(tmp_path))
            is False
        )

    def test_a_missing_source_schema_is_not_current(self, tmp_path: Path) -> None:
        client_dir = self._client_dir(tmp_path, "model Foo {\n  id String @id\n}\n")

        assert (
            prisma_migration.client_already_generated_from(tmp_path / "absent.prisma", client_dir)
            is False
        )

    def test_an_unimportable_prisma_package_is_not_current(self, tmp_path: Path) -> None:
        schema = self._schema(tmp_path, "model Foo {\n  id String @id\n}\n")

        assert prisma_migration.client_already_generated_from(schema, None) is False
