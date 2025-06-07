import pytest
import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.azure.chat.gpt_transformation import AzureOpenAIConfig

@pytest.mark.parametrize("version", [
    "2025-01-01-preview",
    "2023-07-15-preview",
    "1999-12-31-preview",
])
def test_valid_api_versions(version):
    assert AzureOpenAIConfig.is_valid_api_version(version) is True


@pytest.mark.parametrize(
    "api_version,expected",
    [
        ("2025-01-01", True),
        ("2025-01-01-preview", True),
        ("2025-01-01-beta-release", True),
        ("2025-01-01-anything-goes", True),
        ("2025-12-31", True),
        ("2025-12-31-xyz", True),
        ("2025-1-1-preview", False),        # month and day must be 2 digits
        ("25-01-01-preview", False),       # year must be 4 digits
        ("abcd-preview", False),            # not a valid date
        ("", False),                       # empty string
        ("20250101", False),                # no dashes
    ],
)
def test_is_valid_api_version(api_version, expected):
    assert AzureOpenAIConfig.is_valid_api_version(api_version) == expected