from typing import Final

from litellm.rust_bridge.catalog import COMPONENTS
from litellm.rust_bridge.configuration import ComponentName

COMPONENT: Final = COMPONENTS[ComponentName.RERANK]
