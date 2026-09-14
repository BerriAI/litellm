from typing import Final

from litellm.rust_bridge.configuration import RouteName
from litellm.rust_bridge.route import NativeRoute

ROUTE: Final = NativeRoute(RouteName.RESPONSES)
