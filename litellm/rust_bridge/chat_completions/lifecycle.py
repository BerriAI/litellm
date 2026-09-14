from typing import Final

from litellm.rust_bridge.chat_completions import ROUTE

LIFECYCLE: Final = ROUTE.lifecycle()
