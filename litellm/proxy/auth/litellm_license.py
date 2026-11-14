import os
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_proxy_logger
from litellm.constants import NON_LLM_CONNECTION_TIMEOUT
from litellm.rust_bridge._native import Entitlements, LicenseVerifier

if TYPE_CHECKING:
    from litellm.proxy._types import EnterpriseLicenseData

MISSING_AUTO_ROUTER_LICENSE: Final = Entitlements.missing_feature_error("auto_router")


class LicenseCheck:
    """The license key in use and what Rust says it entitles this deployment to"""

    def __init__(self, verifier: LicenseVerifier | None = None) -> None:
        self.license_str = os.getenv("LITELLM_LICENSE", None)
        verbose_proxy_logger.debug("License configured: %s", self.license_str is not None)
        self.verifier: Final = verifier or LicenseVerifier(timeout_seconds=NON_LLM_CONNECTION_TIMEOUT)
        self._premium_check_logged = False
        self.entitlements: Entitlements = Entitlements.COMMUNITY

    @property
    def airgapped_license_data(self) -> "EnterpriseLicenseData | None":
        return self.entitlements.license

    def is_premium(self) -> bool:
        if self.license_str is None:
            self.license_str = os.getenv("LITELLM_LICENSE", None)

        if not self._premium_check_logged:
            verbose_proxy_logger.debug(
                "litellm.proxy.auth.litellm_license.py::is_premium() - License configured after refresh: %s",
                self.license_str is not None,
            )
            self._premium_check_logged = True

        try:
            self.entitlements = self.verifier.entitlements(self.license_str)
        except Exception:
            self.entitlements = Entitlements.COMMUNITY
        premium: Final = self.entitlements.is_enterprise()
        verbose_proxy_logger.debug(
            "litellm.proxy.auth.litellm_license.py::is_premium() - License is premium=%s", premium
        )
        return premium

    def user_seat_limit(self) -> int | None:
        return self.entitlements.max_users

    def is_over_limit(self, total_users: int) -> bool:
        return self.entitlements.is_over_user_limit(total_users)

    def is_team_count_over_limit(self, team_count: int) -> bool:
        return self.entitlements.is_over_team_limit(team_count)

    def auto_router_capability_limit(self) -> int | None:
        """One auto-router per gated capability unless the signed license grants auto_router; None is no limit"""
        return None if self.entitlements.grants("auto_router") else 1
