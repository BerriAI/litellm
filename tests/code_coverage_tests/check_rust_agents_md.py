"""
Keep AGENTS.md files in litellm-rust/crates/llms and llms-types free of duplication and gaps.

1. Each upstream URL appears in exactly one AGENTS.md
2. Every llms/src/<provider>/<format>/ folder and every llms-types format and provider folder has an AGENTS.md
"""

import re
import sys
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parent.parent.parent
CRATES: Final = REPO_ROOT / "litellm-rust" / "crates"
SCANNED: Final = (CRATES / "llms", CRATES / "llms-types")
URL: Final = re.compile(r"https?://[^\s)>`]+")


def agents_files() -> tuple[Path, ...]:
    return tuple(sorted(path for crate in SCANNED for path in crate.rglob("AGENTS.md")))


def urls_in(path: Path) -> frozenset[str]:
    return frozenset(url.rstrip(".,") for url in URL.findall(path.read_text()))


def duplicate_urls(files: tuple[Path, ...]) -> dict[str, tuple[Path, ...]]:
    listed: Final = {path: urls_in(path) for path in files}
    every_url: Final = frozenset(url for urls in listed.values() for url in urls)
    owners: Final = {url: tuple(path for path, urls in listed.items() if url in urls) for url in sorted(every_url)}
    return {url: paths for url, paths in owners.items() if len(paths) > 1}


def required_folders() -> tuple[Path, ...]:
    llms_src: Final = CRATES / "llms" / "src"
    provider_formats: Final = (
        folder
        for provider in llms_src.iterdir()
        if provider.is_dir() and provider.name != "base_llm"
        for folder in provider.iterdir()
        if folder.is_dir()
    )
    types_src: Final = CRATES / "llms-types" / "src"
    type_folders: Final = (
        folder for group in ("formats", "providers") for folder in (types_src / group).iterdir() if folder.is_dir()
    )
    return tuple(sorted((*provider_formats, *type_folders)))


def main() -> int:
    files: Final = agents_files()
    duplicates: Final = duplicate_urls(files)
    missing: Final = tuple(folder for folder in required_folders() if not (folder / "AGENTS.md").exists())
    for url, paths in duplicates.items():
        sys.stdout.write(f"URL listed in more than one AGENTS.md: {url}" + "\n")
        for path in paths:
            sys.stdout.write(f"  {path.relative_to(REPO_ROOT)}" + "\n")
    for folder in missing:
        sys.stdout.write(f"Missing AGENTS.md: {folder.relative_to(REPO_ROOT)}" + "\n")
    if duplicates or missing:
        sys.stdout.write("See the AGENTS.md convention in litellm-rust/crates/llms/AGENTS.md" + "\n")
        return 1
    sys.stdout.write(f"Checked {len(files)} AGENTS.md files" + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
