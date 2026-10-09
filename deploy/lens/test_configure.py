import tempfile
import unittest
from pathlib import Path

from configure import SECRET_NAMES, configure


class ComposeConfigurationTests(unittest.TestCase):
    def test_restart_and_upgrade_preserve_private_credentials_and_custom_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            configure(path, "v1.2.3")
            original = dict(line.split("=", 1) for line in path.read_text().splitlines())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(len({original[name] for name in SECRET_NAMES}), len(SECRET_NAMES))
            self.assertTrue(all(len(original[name]) >= 64 for name in SECRET_NAMES))
            with path.open("a") as output:
                output.write("LITELLM_LENS_PUBLIC_URL=https://traces.example/prefix\n")
            configure(path, "1.2.3")
            configure(path, "v1.2.4-nightly")
            updated = dict(line.split("=", 1) for line in path.read_text().splitlines())
            self.assertEqual({name: updated[name] for name in SECRET_NAMES}, {name: original[name] for name in SECRET_NAMES})
            self.assertEqual(updated["LITELLM_VERSION"], "1.2.4-nightly")
            self.assertEqual(updated["LITELLM_LENS_PUBLIC_URL"], "https://traces.example/prefix")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_incomplete_configuration_is_never_replaced_with_new_database_passwords(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            original = "POSTGRES_PASSWORD=existing\n"
            path.write_text(original)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                configure(path, "1.2.3")
            self.assertEqual(path.read_text(), original)

    def test_invalid_release_and_symlink_leave_existing_files_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            target = Path(directory) / "saved"
            target.write_text("preserve")
            path.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlink"):
                configure(path, "1.2.3")
            self.assertEqual(target.read_text(), "preserve")
            path.unlink()
            with self.assertRaisesRegex(ValueError, "published release"):
                configure(path, "1.2.3\nPOSTGRES_PASSWORD=replaced")
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
