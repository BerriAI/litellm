import sys
from typing import Final

__all__: Final = ("timeout",)

if sys.version_info >= (3, 11):
    from asyncio import timeout
else:
    from async_timeout import timeout
