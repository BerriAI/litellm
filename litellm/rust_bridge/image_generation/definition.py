from typing import Final

from litellm.rust_bridge.catalog import COMPONENTS
from litellm.rust_bridge.configuration import RouteName

COMPONENT: Final = COMPONENTS[RouteName.IMAGE_GENERATION]
