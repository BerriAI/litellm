import pytest

from litellm_provisioning_mcp.naming import (
    RELEASE_MAX_LEN,
    derive_image_repos,
    registry_from_repo_url,
    sanitize_label,
    sanitize_release_name,
)


def test_sanitize_release_name_lowercases_and_replaces_invalid():
    assert sanitize_release_name("Feature/My_Branch") == "feature-my-branch"


def test_sanitize_release_name_truncates_and_strips():
    long = "litellm-e2e-" + "a" * 80
    out = sanitize_release_name(long)
    assert len(out) <= RELEASE_MAX_LEN
    assert not out.endswith("-")


def test_sanitize_release_name_empty_raises():
    with pytest.raises(ValueError):
        sanitize_release_name("///")


def test_sanitize_label_strips_invalid_edges():
    assert sanitize_label("-abc/def-") == "abc-def"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://github.com/BerriAI/litellm", "ghcr.io/berriai"),
        ("https://github.com/BerriAI/litellm.git", "ghcr.io/berriai"),
        ("git@github.com:Alice/litellm.git", "ghcr.io/alice"),
        ("github.com/Bob/litellm", "ghcr.io/bob"),
        ("https://gitlab.com/x/y", None),
    ],
)
def test_registry_from_repo_url(url, expected):
    assert registry_from_repo_url(url) == expected


def test_derive_image_repos_override_wins():
    repos = derive_image_repos(
        repo_url="https://github.com/BerriAI/litellm",
        registry_override="myreg.io/team",
        default_registry="ghcr.io/berriai",
    )
    assert repos["gateway"] == "myreg.io/team/litellm-gateway"
    assert repos["migrations"] == "myreg.io/team/litellm-migrations"


def test_derive_image_repos_derives_from_fork():
    repos = derive_image_repos(
        repo_url="https://github.com/Alice/litellm",
        registry_override=None,
        default_registry="ghcr.io/berriai",
    )
    assert repos["backend"] == "ghcr.io/alice/litellm-backend"


def test_derive_image_repos_falls_back_to_default():
    repos = derive_image_repos(
        repo_url="https://example.com/not-github",
        registry_override=None,
        default_registry="ghcr.io/berriai",
    )
    assert repos["ui"] == "ghcr.io/berriai/litellm-ui"
