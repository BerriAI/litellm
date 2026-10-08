import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Final

REPO: Final = Path(__file__).resolve().parents[2]


class TestUIRootPath(unittest.TestCase):
    def prepare(self, directory: Path, prefix: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "sh",
                str(REPO / "ui" / "prepare-root-path.sh"),
                str(directory / "source"),
                str(directory / "runtime"),
                str(REPO / "ui" / "nginx.conf"),
                str(directory / "nginx.conf"),
                prefix,
            ],
            text=True,
            capture_output=True,
        )

    def source(self, directory: Path) -> Path:
        source: Final = directory / "source"
        source.mkdir()
        (source / "index.html").write_text(
            '<html><head><link href="/get_favicon"><link href="/favicon.ico?version=1"></head><body>'
            '<script src="/litellm-asset-prefix/_next/app.js"></script>'
            '<img src="/ui/assets/logo.png"></body></html>'
        )
        (source / "app.js").write_text('fetch("/litellm/.well-known/litellm-ui-config")')
        (source / "font.woff2").write_bytes(bytes(range(256)))
        return source

    def test_nested_prefix_preserves_original_export_and_binary_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory: Final = Path(tmp)
            source: Final = self.source(directory)
            original: Final = (source / "index.html").read_bytes()
            self.assertEqual(self.prepare(directory, "/services/llm/").returncode, 0)
            runtime: Final = directory / "runtime"
            self.assertIn("/services/llm/_next/app.js", (runtime / "index.html").read_text())
            self.assertIn(
                'name="litellm-server-root-path" content="/services/llm"', (runtime / "index.html").read_text()
            )
            self.assertIn('href="/services/llm/get_favicon"', (runtime / "index.html").read_text())
            self.assertIn('href="/services/llm/favicon.ico?version=1"', (runtime / "index.html").read_text())
            self.assertIn('src="/services/llm/ui/assets/logo.png"', (runtime / "index.html").read_text())
            self.assertEqual(runtime.stat().st_mode & 0o005, 0o005)
            self.assertEqual((runtime / "index.html").stat().st_mode & 0o004, 0o004)
            self.assertIn("/services/llm/.well-known/litellm-ui-config", (runtime / "app.js").read_text())
            self.assertEqual((source / "index.html").read_bytes(), original)
            self.assertEqual((runtime / "font.woff2").read_bytes(), (source / "font.woff2").read_bytes())
            self.assertEqual(self.prepare(directory, "/services/llm/").returncode, 0)
            self.assertNotIn("/services/llm/services/llm", (runtime / "app.js").read_text())
            self.assertEqual(self.prepare(directory, "/other/path").returncode, 0)
            self.assertIn("/other/path/_next/app.js", (runtime / "index.html").read_text())
            self.assertNotIn("/services/llm", (runtime / "index.html").read_text())

    def test_default_root_leaves_export_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory: Final = Path(tmp)
            source: Final = self.source(directory)
            self.assertEqual(self.prepare(directory, "/").returncode, 0)
            self.assertEqual((directory / "runtime" / "app.js").read_bytes(), (source / "app.js").read_bytes())

    def test_rejects_unsafe_roots_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory: Final = Path(tmp)
            self.source(directory)
            for prefix in ("relative", "/a/../b", "/a/./b", "/a//b", "/a|b", "/a?b", "/a\nb", "//", "/a//"):
                with self.subTest(prefix=prefix):
                    self.assertNotEqual(self.prepare(directory, prefix).returncode, 0)
                    self.assertFalse((directory / "runtime").exists())

    def test_source_can_be_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory: Final = Path(tmp)
            source: Final = self.source(directory)
            (source / "app.js").chmod(0o444)
            digest: Final = hashlib.sha256((source / "app.js").read_bytes()).hexdigest()
            self.assertEqual(self.prepare(directory, "/services/llm").returncode, 0)
            self.assertEqual(hashlib.sha256((source / "app.js").read_bytes()).hexdigest(), digest)
