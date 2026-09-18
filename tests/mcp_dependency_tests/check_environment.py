from collections.abc import Iterable
import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path
import sys
import sysconfig
from typing import Final
import unittest


from packaging.utils import canonicalize_name


def installed_versions(distributions: Iterable[importlib.metadata.Distribution]) -> dict[str, str]:
    return {canonicalize_name(distribution.metadata["Name"]): distribution.version for distribution in distributions}


def main(profile: str, environment: Path) -> None:
    import litellm

    package: Final = Path(litellm.__file__).resolve()
    assert package.is_relative_to(environment.resolve()), f"wrong wheel import: {package}"
    installed: Final = installed_versions(importlib.metadata.distributions())
    if profile == "core":
        assert all(importlib.util.find_spec(name) is None for name in ("mcp", "mcp_types", "httpx2", "httpcore2"))
    else:
        import httpx
        import httpx2
        import mcp
        from mcp.types import Tool
        from pydantic import ValidationError

        assert installed["mcp"] == "2.2.0"
        assert tuple(int(part) for part in installed["httpx2"].split(".")[:2]) >= (2, 12)
        assert httpx.AsyncClient is not httpx2.AsyncClient
        assert Path(mcp.__file__).resolve().is_relative_to(environment.resolve())
        tool: Final = Tool.model_validate({"name": "echo", "inputSchema": {"type": "object"}})
        encoded: Final = tool.model_dump(by_alias=True, exclude_none=True)
        assert encoded["inputSchema"] == {"type": "object"}
        assert Tool.model_validate(encoded) == tool
        with unittest.TestCase().assertRaises(ValidationError) as failure:
            Tool.model_validate({"inputSchema": {"type": "object"}})
        assert any(item["loc"] == ("name",) for item in failure.exception.errors())
    report: Final = {
        "profile": profile,
        "python": sys.version,
        "litellm_path": str(package),
        "installed": installed,
        "environment": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "python_full_version": platform.python_version(),
            "sys_platform": sys.platform,
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "implementation_name": sys.implementation.name,
            "platform_python_implementation": platform.python_implementation(),
            "extra": "",
        },
        "site_packages_bytes": sum(
            path.stat().st_size for path in Path(sysconfig.get_path("purelib")).rglob("*") if path.is_file()
        ),
    }
    (environment / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
