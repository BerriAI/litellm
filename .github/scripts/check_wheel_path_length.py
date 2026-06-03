"""Guard against shipping wheel paths long enough to break the Windows 260-char
MAX_PATH limit at install time.

litellm has repeatedly shipped content-filter benchmark *test fixtures* with very
long names under
``litellm/proxy/guardrails/guardrail_hooks/litellm_content_filter/guardrail_benchmarks/``.
On a default Windows machine (long-path support off, the OS default) ``pip install
litellm`` then aborts mid-unpack with ``OSError: [Errno 2] No such file or directory``,
leaving a half-installed package (no ``litellm.types``) -> ``ModuleNotFoundError``.
See issues/PRs #21941, #22039, #29536, #29553 -- it has regressed three times.

The path that actually lands on disk is ``<site-packages prefix> + <path inside the
wheel>``. A realistic worst-case Windows prefix (long profile name + roaming AppData
venv), e.g.::

    C:\\Users\\Administrator\\AppData\\Roaming\\<app>\\<sub>\\venv\\Lib\\site-packages\\   (~95 chars)

so we budget ``260 - 100 = 160`` chars for any single path inside the wheel. This is
measured on the BUILT wheel, so it honours ``[tool.uv.build-backend].source-exclude``
(i.e. it passes once excluded fixtures are no longer packaged).
"""
import glob
import os
import sys
import zipfile

# Windows MAX_PATH (260) minus ~100 chars for a realistic install prefix.
MAX_RELATIVE = 160


def main(dist_dir: str) -> int:
    wheels = glob.glob(os.path.join(dist_dir, "*.whl"))
    if not wheels:
        print(f"::error::no .whl found in {dist_dir!r}")
        return 1

    rc = 0
    for whl in wheels:
        names = zipfile.ZipFile(whl).namelist()
        longest = max((len(n) for n in names), default=0)
        offenders = sorted(
            (n for n in names if len(n) > MAX_RELATIVE), key=len, reverse=True
        )
        print(f"{os.path.basename(whl)}: {len(names)} entries, longest path = {longest}")
        if offenders:
            rc = 1
            print(
                f"::error::{len(offenders)} packaged path(s) exceed {MAX_RELATIVE} chars "
                f"and risk the Windows MAX_PATH limit at install time:"
            )
            for n in offenders[:15]:
                print(f"  {len(n):4}  {n}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "dist"))
