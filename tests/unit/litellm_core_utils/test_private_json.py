import json
import os
import stat

import pytest

from litellm.litellm_core_utils.private_json import (
    overwrite_private_json,
    write_private_bytes,
    write_private_json,
)


class TestOverwritePrivateJson:
    def test_replaces_the_contents_of_the_file_already_there(self, tmp_path):
        path = tmp_path / "token.json"
        write_private_json(str(path), {"key": "sk-" + "a" * 700})

        overwrite_private_json(str(path), {"user_id": "u-1"})

        assert json.loads(path.read_text()) == {"user_id": "u-1"}

    def test_refuses_to_create_the_file_it_was_asked_to_rewrite(self, tmp_path):
        """This is the one writer that does not go through a private temp file, so a path it creates
        would land with whatever the umask allows. Refusing keeps it unable to put a world-readable
        file where the caller believed a private one already was."""
        path = tmp_path / "token.json"

        with pytest.raises(FileNotFoundError):
            overwrite_private_json(str(path), {"user_id": "u-1"})

        assert not path.exists()

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
    def test_keeps_the_owner_only_mode_the_file_was_created_with(self, tmp_path):
        path = tmp_path / "token.json"
        write_private_json(str(path), {"key": "sk-live"})

        overwrite_private_json(str(path), {"user_id": "u-1"})

        assert stat.S_IMODE(path.stat().st_mode) == 0o600


class TestWritePrivateBytes:
    def test_replaces_the_file_in_one_step_so_a_reader_holding_the_old_one_keeps_it_whole(self, tmp_path):
        path = tmp_path / "script.py"
        write_private_bytes(str(path), b"print('one')\n" * 200)
        before = path.stat().st_ino

        with path.open("rb") as reader:
            write_private_bytes(str(path), b"print('two')\n")
            assert reader.read() == b"print('one')\n" * 200

        assert path.read_bytes() == b"print('two')\n"
        assert path.stat().st_ino != before
        assert [child.name for child in tmp_path.iterdir()] == ["script.py"]

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
    def test_lands_owner_only_and_a_refused_stage_leaves_the_previous_file_untouched(self, tmp_path):
        path = tmp_path / "script.py"
        write_private_bytes(str(path), b"first")
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

        tmp_path.chmod(0o500)
        try:
            with pytest.raises(PermissionError):
                write_private_bytes(str(path), b"second")
        finally:
            tmp_path.chmod(0o700)
        assert path.read_bytes() == b"first"
