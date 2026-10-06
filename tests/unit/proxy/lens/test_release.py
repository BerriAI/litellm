from collections.abc import Mapping, Sequence
from importlib.metadata import Distribution, PackageNotFoundError, PathDistribution
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

from litellm.proxy.lens.release import worker_image


@pytest.mark.parametrize("tag", ("v1.2.3", "v1.2.3-rc.4", "v1.2.3-dev.5", "branch-main-1234567"))
def test_install_command_follows_the_gateway_release(monkeypatch: pytest.MonkeyPatch, tag: str) -> None:
    monkeypatch.setenv("LITELLM_RELEASE_TAG", tag)
    monkeypatch.delenv("LENS_WORKER_IMAGE", raising=False)
    assert worker_image() == f"ghcr.io/berriai/litellm-lens-worker:{tag}"


def test_private_registry_override_keeps_its_exact_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    image: Final = "registry.example/lens-worker@sha256:" + "a" * 64
    monkeypatch.setenv("LITELLM_RELEASE_TAG", "branch-main-1234567")
    monkeypatch.setenv("LENS_WORKER_IMAGE", image)
    assert worker_image() == image


def test_source_build_uses_the_separate_development_package(monkeypatch: pytest.MonkeyPatch) -> None:
    tag: Final = "sha-" + "a" * 40
    monkeypatch.setenv("LITELLM_RELEASE_TAG", tag)
    monkeypatch.delenv("LENS_WORKER_IMAGE", raising=False)
    assert worker_image() == f"ghcr.io/berriai/litellm-lens-worker-dev:{tag}"


@pytest.mark.parametrize(
    "installed,expected",
    (("1.2.3", "v1.2.3"), ("1.2.3rc4", "v1.2.3-rc.4"), ("1.2.3.dev5", "v1.2.3-dev.5")),
)
def test_python_installs_recommend_the_matching_worker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, installed: str, expected: str
) -> None:
    from litellm.proxy.lens import release

    metadata: Final = tmp_path / "litellm.dist-info"
    metadata.mkdir()
    metadata.joinpath("METADATA").write_text(f"Name: litellm\nVersion: {installed}\n")

    def installed_distribution(name: str) -> Distribution:
        assert name in ("litellm-core", "litellm")
        return PathDistribution(metadata)

    monkeypatch.delenv("LITELLM_RELEASE_TAG", raising=False)
    monkeypatch.delenv("LENS_WORKER_IMAGE", raising=False)
    monkeypatch.setattr(release, "distribution", installed_distribution)
    monkeypatch.setattr(release, "__file__", str(tmp_path / "litellm/proxy/lens/release.py"))
    assert release.release_tag() == expected
    assert worker_image() == f"ghcr.io/berriai/litellm-lens-worker:{expected}"


@pytest.mark.parametrize("source", ("checkout", "direct-install", "unversioned-container", "missing-package"))
def test_unknown_source_never_falls_back_to_a_package_version_or_image_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: str
) -> None:
    from litellm.proxy.lens import release

    metadata: Final = tmp_path / "litellm.dist-info"
    metadata.mkdir()
    metadata.joinpath("METADATA").write_text("Name: litellm\nVersion: 1.2.3\n")
    if source == "direct-install":
        metadata.joinpath("direct_url.json").write_text('{"url":"file:///checkout","dir_info":{"editable":true}}')

    def installed_distribution(name: str) -> Distribution:
        if source == "missing-package":
            raise PackageNotFoundError(name)
        return PathDistribution(metadata)

    monkeypatch.delenv("LITELLM_RELEASE_TAG", raising=False)
    monkeypatch.setenv("LENS_WORKER_IMAGE", "registry.example/lens-worker:old")
    monkeypatch.setattr(release, "distribution", installed_distribution)
    if source != "checkout":
        monkeypatch.setattr(release, "__file__", str(tmp_path / "litellm/proxy/lens/release.py"))
    if source == "unversioned-container":
        monkeypatch.setenv("LITELLM_RELEASE_TAG", "")
    assert release.release_tag() == ""
    assert worker_image() == ""


@pytest.mark.parametrize("tag,missing", (("sha-" + "a" * 40, "litellm"), ("", "litellm"), ("", "broken_sdk_dependency")))
def test_standalone_worker_release_does_not_require_the_sdk(
    monkeypatch: pytest.MonkeyPatch, tag: str, missing: str
) -> None:
    import builtins
    import runpy

    from litellm.proxy.lens import release

    original_import: Final = builtins.__import__

    def without_sdk(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        if name.startswith("litellm"):
            raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", without_sdk)
    if tag:
        monkeypatch.setenv("LITELLM_RELEASE_TAG", tag)
    else:
        monkeypatch.delenv("LITELLM_RELEASE_TAG", raising=False)
    namespace: Final = runpy.run_path(release.__file__)
    if missing == "litellm":
        assert namespace["release_tag"]() == tag
    else:
        with pytest.raises(ModuleNotFoundError) as caught:
            namespace["release_tag"]()
        assert caught.value.name == missing
