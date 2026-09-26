from typing import Final

from litellm.proxy.db.queries import load

SET_LOCK_TIMEOUT: Final = load(__name__, "set_lock_timeout")
SET_STATEMENT_TIMEOUT: Final = load(__name__, "set_statement_timeout")
