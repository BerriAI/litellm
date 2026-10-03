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


@pytest.mark.parametrize(
    "installed,expected",
    (("1.2.3", "v1.2.3"), ("1.2.3rc4", "v1.2.3-rc.4"), ("1.2.3.dev5", "v1.2.3-dev.5")),
)
def test_python_installs_recommend_the_matching_worker(
    monkeypatch: pytest.MonkeyPatch, installed: str, expected: str
) -> None:
    from litellm.proxy.lens import release

    def installed_version(name: str) -> str:
        assert name == "litellm"
        return installed

    monkeypatch.delenv("LITELLM_RELEASE_TAG", raising=False)
    monkeypatch.delenv("LENS_WORKER_IMAGE", raising=False)
    monkeypatch.setattr(release, "version", installed_version)
    assert release.release_tag() == expected
    assert worker_image() == f"ghcr.io/berriai/litellm-lens-worker:{expected}"
