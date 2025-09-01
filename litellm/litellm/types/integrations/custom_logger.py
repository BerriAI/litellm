from typing import Optional

from pydantic import BaseModel


class StandardCustomLoggerInitParams(BaseModel):
    """
    Params for initializing a CustomLogger.
    """
    turn_off_message_logging: Optional[bool] = False