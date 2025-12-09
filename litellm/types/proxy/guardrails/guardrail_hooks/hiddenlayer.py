import enum


class HiddenlayerAction(str, enum.Enum):
    BLOCK = "Block"
    REDACT = "Redact"


class HiddenlayerMessages(str, enum.Enum):
    BLOCK_MESSAGE = "Blocked by Hiddenlayer."
