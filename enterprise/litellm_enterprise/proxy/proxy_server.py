from typing import Optional

from litellm_enterprise.types.proxy.proxy_server import CustomAuthSettings

custom_auth_settings: Optional[CustomAuthSettings] = None


class EnterpriseProxyConfig:
    async def load_custom_auth_settings(
        self, general_settings: dict
    ) -> CustomAuthSettings:
        print(f"General settings: {general_settings}")
        custom_auth_settings = general_settings.get("custom_auth_settings", None)
        print(f"Custom auth settings: {custom_auth_settings}")
        if custom_auth_settings is not None:
            custom_auth_settings = CustomAuthSettings(
                mode=custom_auth_settings.get("mode"),
            )
        print(f"Custom auth settings: {custom_auth_settings}")
        return custom_auth_settings

    async def load_enterprise_config(self, general_settings: dict) -> None:
        global custom_auth_settings
        custom_auth_settings = await self.load_custom_auth_settings(general_settings)
        return None
