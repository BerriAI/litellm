import os

import pytest

from litellm.litellm_core_utils.path_utils import is_within, join_within, safe_filename, safe_join, try_safe_join


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
        with pytest.raises(ValueError, match="Empty or unsafe filename"):
            safe_filename("")


def test_try_safe_join_returns_none_instead_of_raising(tmp_path):
    inside = try_safe_join(str(tmp_path), "categories", "x.yaml")
    assert inside is not None and inside.startswith(os.path.realpath(str(tmp_path)))
    assert try_safe_join(str(tmp_path), "..", "escaped.yaml") is None
    assert try_safe_join(str(tmp_path), "bad\x00name") is None


def test_is_within_resolves_symlinks_before_checking(tmp_path):
    outside = tmp_path / "outside.yaml"
    outside.write_text("x")
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "inside.yaml").write_text("x")
    (folder / "out_link.yaml").symlink_to(outside)
    (folder / "in_link.yaml").symlink_to(folder / "inside.yaml")

    assert is_within(str(folder / "inside.yaml"), str(folder))
    assert is_within(str(folder / "in_link.yaml"), str(folder))
    assert is_within(str(folder), str(folder))
    assert not is_within(str(folder / "out_link.yaml"), str(folder))
    assert not is_within(str(folder / ".." / "outside.yaml"), str(folder))
    assert not is_within(str(tmp_path / "folder_sibling.yaml"), str(folder))


def test_join_within_keeps_symlinks_but_rejects_traversal(tmp_path):
    outside = tmp_path / "outside.yaml"
    outside.write_text("x")
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "link.yaml").symlink_to(outside)

    kept = join_within(str(folder), "link.yaml")
    assert kept == os.path.join(os.path.normpath(os.path.abspath(str(folder))), "link.yaml")
    assert os.path.islink(kept)
    assert join_within(str(folder), "..", "outside.yaml") is None
    assert join_within(str(folder), "sub", "..", "..", "outside.yaml") is None
    assert join_within(str(folder), str(outside)) is None
    assert join_within(str(folder), "bad\x00name") is None
    with pytest.raises(ValueError, match="escapes base directory"):
        safe_join(str(folder), "link.yaml")
