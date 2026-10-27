from litellm.integrations.otel.mappers.base import AttributeMap, SpanData
from litellm.integrations.otel.mappers.utils import drop_none_pairs
from litellm.integrations.otel.model.payloads import (
    LLMCallSpanData,
    MCPListToolsSpanData,
    MCPToolCallSpanData,
)
from litellm.integrations.otel.model.semconv import User


class MixpanelMapper:
    def map(self, data: SpanData) -> AttributeMap:
        match data:
            case LLMCallSpanData():
                return drop_none_pairs(((User.ID, data.identity.user_id or data.trace.user_id or None),))
            case MCPToolCallSpanData() | MCPListToolsSpanData():
                return drop_none_pairs(((User.ID, data.identity.user_id or None),))
            case _:
                return drop_none_pairs(())
