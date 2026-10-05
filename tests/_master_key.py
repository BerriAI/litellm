import secrets
from typing import Final

MASTER_KEY: Final = f"sk-{secrets.token_hex(16)}"
