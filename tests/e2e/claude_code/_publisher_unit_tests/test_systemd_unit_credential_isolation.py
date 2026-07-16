"""Pin: the cron systemd unit hides credential-bearing dotdirs.

`ProtectHome=read-only` blocks writes to /home/mateo but still allows
reads. A model-directed `Read` tool call (the PDF cells pass
`--allowed-tools Read` to the `claude` CLI) or a compromised
`@anthropic-ai/claude-code` package can read absolute paths under
the runtime user's home and exfiltrate the contents — even with the
per-`claude`-invocation HOME isolation in place, because absolute
paths bypass `~`-expansion.

This file pins the second line of defense: the systemd unit lists
the credential-bearing dotdirs (`~/.config/gh`, `~/.ssh`, `~/.aws`,
`~/.docker`, `~/.kube`, `~/.gnupg`) under `InaccessiblePaths=` so
the kernel hides them from every process in the unit's mount
namespace, including any child of `claude --version` or the pytest
run. It also pins that `~/.config/gh` is *not* in `ReadWritePaths=`
— we pass `GH_TOKEN` inline to every `gh` invocation in
`run_daily.sh`, so the host gh-cli config is unused.
"""

from __future__ import annotations

import re
from pathlib import Path

SERVICE = (
    Path(__file__).resolve().parents[1] / "cron_vm" / "litellm-compat-matrix.service"
)


def _service_text() -> str:
    return SERVICE.read_text()


def _directive(name: str) -> str:
    """Return the value of a single-line systemd directive (or empty)."""
    text = _service_text()
    match = re.search(rf"^\s*{re.escape(name)}\s*=\s*(.*)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def test_inaccessible_paths_hides_credential_dotdirs() -> None:
    """Every credential-bearing dotdir must be under `InaccessiblePaths=`."""
    inaccessible = _directive("InaccessiblePaths")
    assert inaccessible, (
        "litellm-compat-matrix.service: must declare `InaccessiblePaths=` "
        "to hide credential dotdirs from the `claude` subprocess and the "
        "model-directed Read tool. Without this, an absolute-path read "
        "like `Read('/home/mateo/.config/gh/hosts.yml')` exfiltrates "
        "the gh-cli token despite the per-invocation HOME isolation."
    )
    for path in (
        "/home/mateo/.config/gh",
        "/home/mateo/.ssh",
        "/home/mateo/.aws",
        "/home/mateo/.docker",
        "/home/mateo/.kube",
        "/home/mateo/.gnupg",
    ):
        # Tolerated `-` prefix means "ignore if missing on host".
        assert path in inaccessible, (
            f"litellm-compat-matrix.service: `{path}` must appear in "
            f"`InaccessiblePaths=` so the cron `claude` subprocess can "
            f"never read it (even via an absolute path that bypasses "
            f"the per-invocation HOME override)."
        )


def test_gh_config_is_not_writeable() -> None:
    """`~/.config/gh` is not whitelisted under `ReadWritePaths=`.

    We pass `GH_TOKEN` inline to every `gh` invocation in
    `run_daily.sh` (`gh repo clone`, `gh pr create`, `gh pr edit`).
    The host `~/.config/gh/hosts.yml` is therefore never consulted
    or written to. Keeping it out of `ReadWritePaths=` is the second
    line of defense: a future regression that drops the inline-token
    convention will fail loudly (gh writes a new login config and
    hits a read-only filesystem) rather than silently re-introduce
    the credential exfiltration surface that
    `InaccessiblePaths=/home/mateo/.config/gh` is closing.
    """
    rw = _directive("ReadWritePaths")
    assert ".config/gh" not in rw, (
        "litellm-compat-matrix.service: `/home/mateo/.config/gh` must "
        "*not* appear in `ReadWritePaths=`. We pass `GH_TOKEN` inline "
        "to every `gh` invocation in run_daily.sh, so the host gh-cli "
        "config is never consulted or written to. Keeping the path out "
        "of ReadWritePaths means a future regression that drops the "
        "inline-token convention will fail loudly instead of silently "
        "re-opening the credential exfiltration surface that "
        "`InaccessiblePaths=` is closing."
    )


def test_protect_home_is_read_only_or_stricter() -> None:
    """`ProtectHome=` must be at least `read-only`."""
    value = _directive("ProtectHome")
    assert value in ("read-only", "tmpfs", "yes", "true"), (
        f"litellm-compat-matrix.service: `ProtectHome=` must be `read-only`, "
        f"`tmpfs`, or `yes`. Got: {value!r}. Without this, the unit can "
        f"write anywhere under /home/mateo, including overwriting "
        f"~/.config/gh/hosts.yml."
    )
