from litellm.llms.volcengine.chat.transformation import VolcEngineChatConfig


class BytePlusChatConfig(VolcEngineChatConfig):
    """
    BytePlus chat configuration.
    Uses the BytePlus Southeast Asia endpoint: https://ark.ap-southeast.bytepluses.com/api/v3
    API key: BYTEPLUS_API_KEY

    BytePlus is the international version of Volcengine (ByteDance).
    """

    pass
