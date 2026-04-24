import os

import pytest

from litellm.proxy.common_utils.path_utils import safe_filename, safe_join


class TestSafeJoin:
    def test_normal_path(self, tmp_path):
        result = safe_join(str(tmp_path), "subdir", "file.yaml")
        assert result == os.path.join(str(tmp_path), "subdir", "file.yaml")

    def test_traversal_blocked(self, tmp_path):
        with pytest.raises(ValueError, match="escapes base directory"):
            safe_join(str(tmp_path), "../../etc/passwd.yaml")

    def test_null_byte_blocked(self, tmp_path):
        with pytest.raises(ValueError, match="null byte"):
            safe_join(str(tmp_path), "file\x00.yaml")

    def test_base_dir_itself(self, tmp_path):
        result = safe_join(str(tmp_path))
        assert result == str(tmp_path.resolve())


class TestSafeFilename:
    def test_normal_filename(self):
        assert safe_filename("document.prompt") == "document.prompt"

    def test_strips_unix_path(self):
        assert safe_filename("../../etc/passwd.prompt") == "passwd.prompt"

    def test_strips_windows_path(self):
        assert safe_filename("..\\..\\etc\\passwd.prompt") == "passwd.prompt"

    def test_null_byte_blocked(self):
        with pytest.raises(ValueError, match="null byte"):
            safe_filename("file\x00.prompt")

    def test_dotdot_rejected(self):
        with pytest.raises(ValueError, match="unsafe filename"):
            safe_filename("..")

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            safe_filename("")
