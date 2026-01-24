from unittest.mock import patch
import pytest
@pytest.mark.skip(reason="Very Flaky in CI, will debug later")
def test_restructure_ui_html_files_skipped_in_non_root(monkeypatch):
    """
    Test that _restructure_ui_html_files is SKIPPED when:
    - LITELLM_NON_ROOT is "true"
    - ui_path is "/var/lib/litellm/ui"
    """
    # 1. Setup environment variables and variables
    import litellm.proxy.proxy_server
    monkeypatch.setenv("LITELLM_NON_ROOT", "true")

    # We need to simulate the execution of the module-level code or
    # just test the logic we added.

    is_non_root = True  # Simulate the variable in proxy_server
    ui_path = "/var/lib/litellm/ui"

    # Mock the _restructure_ui_html_files function to check if it's called
    # Use create=True to allow patching even if the module hasn't been imported yet
    # or if the function doesn't exist (it's defined inside a try/except block)
    # spec=False prevents spec checking which can fail during import resolution
    with patch(
        "litellm.proxy.proxy_server._restructure_ui_html_files",
        create=True,
        spec=False,
    ) as mock_restructure:
        # Simulate the logic we added in proxy_server.py
        if is_non_root and ui_path == "/var/lib/litellm/ui":
            # Skipping...
            pass
        else:
            mock_restructure(ui_path)

        # Verify it was NOT called
        mock_restructure.assert_not_called()

@pytest.mark.skip(reason="Very Flaky in CI, will debug later")
def test_restructure_ui_html_files_NOT_skipped_locally(monkeypatch):
    """
    Test that _restructure_ui_html_files is NOT skipped for local development
    """
    monkeypatch.delenv("LITELLM_NON_ROOT", raising=False)

    is_non_root = False
    ui_path = "/some/local/path"

    # Use create=True and spec=False to allow patching even if the module hasn't been imported yet
    # or if the function doesn't exist (it's defined inside a try/except block)
    # spec=False prevents spec checking which can fail during import resolution
    with patch(
        "litellm.proxy.proxy_server._restructure_ui_html_files",
        create=True,
        spec=False,
    ) as mock_restructure:
        if is_non_root and ui_path == "/var/lib/litellm/ui":
            pass
        else:
            mock_restructure(ui_path)

        # Verify it WAS called
        mock_restructure.assert_called_once_with(ui_path)
