from typing import Any, Dict, Optional

from litellm.types.utils import CallTypes

from ..integrations.custom_logger import CustomLogger


class ForwardClientSideHeadersByModelGroup(CustomLogger):
    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Optional[CallTypes]
    ) -> Optional[dict]:
        return None
