from litellm.llms.volcengine.chat.transformation import VolcEngineChatConfig


class VolcEnginePlanChatConfig(VolcEngineChatConfig):
    """
    Volcengine Plan chat configuration.
    Uses the /api/coding/v3 endpoint instead of /api/v3.
    Shares API key (VOLCENGINE_API_KEY / ARK_API_KEY) with the volcengine provider.

    Reference: https://www.volcengine.com/docs/82379/1494384
    """

    pass
