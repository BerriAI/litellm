import os
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.constants import NON_LLM_CONNECTION_TIMEOUT
from litellm.rust_bridge._native import License, LicenseVerifier

__all__ = ("AUTO_ROUTER_LICENSE_FEATURE", "AUTO_ROUTER_LICENSE_REMEDY", "License", "LicenseCheck", "LicenseVerifier")

AUTO_ROUTER_LICENSE_FEATURE: Final = "auto_router"
AUTO_ROUTER_LICENSE_REMEDY: Final = "A LiteLLM license with the 'auto_router' feature lifts the limit."


class LicenseCheck:
    """
    - Check if license in env
    - Returns if license is valid
    """

    def __init__(self, verifier: LicenseVerifier | None = None) -> None:
        self.license_str = os.getenv("LITELLM_LICENSE", None)
        verbose_proxy_logger.debug("License configured: %s", self.license_str is not None)
        self.verifier: Final = verifier or LicenseVerifier(timeout_seconds=NON_LLM_CONNECTION_TIMEOUT)
        self._premium_check_logged = False
        self.airgapped_license_data: License | None = None

    def _verify(self, license_str: str) -> bool:
        premium: Final = self.verifier.verify_remote(license_str)
        verbose_proxy_logger.debug("litellm.proxy.auth.litellm_license.py::_verify - License is premium=%s", premium)
        return premium

    def is_premium(self) -> bool:
        """
        1. verify_license_without_api_request: checks if license was generate using private / public key pair
        2. _verify: checks if license is valid calling litellm API. This is the old way we were generating/validating license
        """
        try:
            if not self._premium_check_logged:
                verbose_proxy_logger.debug(
                    "litellm.proxy.auth.litellm_license.py::is_premium() - ENTERING 'IS_PREMIUM' - License configured: %s",
                    self.license_str is not None,
                )

            if self.license_str is None:
                self.license_str = os.getenv("LITELLM_LICENSE", None)

            if not self._premium_check_logged:
                verbose_proxy_logger.debug(
                    "litellm.proxy.auth.litellm_license.py::is_premium() - License configured after refresh: %s",
                    self.license_str is not None,
                )
                self._premium_check_logged = True

            if self.license_str is None:
                return False
            return self.verify_license_without_api_request(self.license_str) or self._verify(self.license_str)
        except Exception:
            return False

    def user_seat_limit(self) -> int | None:
        return None if self.airgapped_license_data is None else self.airgapped_license_data.max_users

    def is_over_limit(self, total_users: int) -> bool:
        return self.airgapped_license_data is not None and self.airgapped_license_data.is_over_user_limit(total_users)

    def is_team_count_over_limit(self, team_count: int) -> bool:
        return self.airgapped_license_data is not None and self.airgapped_license_data.is_over_team_limit(team_count)

    def grants_feature(self, feature: str) -> bool:
        return self.airgapped_license_data is not None and self.airgapped_license_data.grants(feature)

    def auto_router_capability_limit(self) -> int | None:
        """
        How many auto-routers may claim each gated classifier or customization capability:
        unlimited (None) only when the signed license lists the auto_router feature or the
        "*" wildcard that grants every feature, otherwise one per capability. A license verified
        through the API carries no feature list, so it does not lift the limit either.
        """
        if self.grants_feature(AUTO_ROUTER_LICENSE_FEATURE):
            return None
        return 1

    def verify_license_without_api_request(self, license_key: str) -> bool:
        self.airgapped_license_data = self.verifier.verify_offline(license_key)
        return self.airgapped_license_data is not None
