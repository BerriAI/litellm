"""Run the normal single-process CLI with the existing behavior-suite test entitlement."""

from unittest.mock import patch

from litellm import run_server


def main() -> None:
    with patch(  # test-quality-ok: route entitlement only; license validation is outside these HTTP/DB contracts
        "litellm.proxy.auth.litellm_license.LicenseCheck.is_premium", return_value=True
    ):
        run_server()


if __name__ == "__main__":
    main()
