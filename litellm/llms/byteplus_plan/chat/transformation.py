from litellm.llms.volcengine.chat.transformation import VolcEngineChatConfig


class BytePlusPlanChatConfig(VolcEngineChatConfig):
    """
    BytePlus Plan chat configuration.
    Uses the BytePlus Southeast Asia coding endpoint: https://ark.ap-southeast.bytepluses.com/api/coding/v3
    Shares API key (BYTEPLUS_API_KEY) with the byteplus provider.

    BytePlus is the international version of Volcengine (ByteDance).
    """

    pass
