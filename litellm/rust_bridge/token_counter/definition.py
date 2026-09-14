from typing import Final

from litellm.rust_bridge.catalog import COMPONENTS
from litellm.rust_bridge.configuration import UtilityName

COMPONENT: Final = COMPONENTS[UtilityName.TOKEN_COUNTER]
