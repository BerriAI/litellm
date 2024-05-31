# What is this?
## If litellm license in env, checks if it's valid
import os
from litellm.llms.custom_httpx.http_handler import HTTPHandler


class LicenseCheck:
    """
    - Check if license in env
    - Returns if license is valid
    """

    base_url = "https://license.litellm.ai"

    def __init__(self) -> None:
        self.license_str = os.getenv("LITELLM_LICENSE", None)
        self.http_handler = HTTPHandler()

    def _verify(self, license_str: str) -> bool:
        url = "{}/verify_license/{}".format(self.base_url, license_str)

        try:  # don't impact user, if call fails
            response = self.http_handler.get(url=url)

            response.raise_for_status()

            response_json = response.json()

            premium = response_json["verify"]

            assert isinstance(premium, bool)

            return premium
        except Exception as e:
            return False

    def is_premium(self) -> bool:
        try:
            if self.license_str is None:
                return False
            elif self._verify(license_str=self.license_str):
                return True
            return False
        except Exception as e:
            return False
