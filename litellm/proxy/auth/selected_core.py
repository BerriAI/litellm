from typing import Final

from litellm.proxy.auth.core.core import AuthCore
from litellm.proxy.auth.core.python_core import PythonCore

AUTH_CORE: Final[AuthCore] = PythonCore()
