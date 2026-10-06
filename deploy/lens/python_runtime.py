import json
import subprocess
import sys
import sysconfig
from itertools import chain
from pathlib import Path
from typing import Final


def dependencies(path: Path, loader: Path) -> tuple[Path, ...]:
    result: Final = subprocess.run((str(loader), "--list", str(path)), capture_output=True, text=True, check=True)
    if "not found" in result.stdout:
        raise RuntimeError(f"Missing Python runtime library: {path}")
    words: Final = tuple(result.stdout.split())
    return tuple(Path(word).resolve() for word in words if word.startswith("/"))


def main() -> None:
    stdlib: Final = Path(sysconfig.get_path("stdlib")).resolve()
    executable: Final = Path(sys.executable).resolve()
    loaders: Final = tuple(Path("/usr/lib").glob("ld-linux-*.so.*"))
    if len(loaders) != 1:
        raise RuntimeError("Expected one native glibc dynamic loader in the Lens worker image")
    entries: Final = tuple(
        path for path in stdlib.iterdir() if path.name not in ("site-packages", "dist-packages", "__pycache__")
    )
    extensions: Final = tuple((stdlib / "lib-dynload").glob("*.so"))
    libraries: Final = frozenset(
        chain.from_iterable(dependencies(binary, loaders[0]) for binary in (executable, *extensions))
    )
    manifest: Final = {
        "executable": str(executable),
        "directories": (str(stdlib),),
        "read": tuple(sorted(str(path) for path in {*entries, *libraries})),
        "execute": tuple(sorted(str(path) for path in (executable, *loaders))),
    }
    Path(sys.argv[1]).write_text(json.dumps(manifest), encoding="utf-8")


if __name__ == "__main__":
    main()
