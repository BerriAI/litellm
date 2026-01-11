import os
import uuid
from datetime import datetime


def boolean(key, x: str) -> bool:
    return x.lower() in {"true", "1", "yes", "on"}


def positive_int(key, x: str) -> int:
    try:
        y = int(x)
    except ValueError:
        raise ValueError(f"{key} must be an integer, got: {x}")

    if y < 0:
        raise ValueError(f"{key} must be positive, got: {x}")

    return y


def get_env(key, default=None, validate=None, required=False):
    x = os.getenv(key, "")

    if not x and not default and required:
        raise ValueError(f"{key} must be set")

    if x and validate:
        return validate(key, x)

    return default if not x else x


def make_key(timestamp: datetime, path: str | None) -> str:
    base_path = f"{path}/" if path else ""

    # keys are partitioned by minute
    date_part = timestamp.strftime("%Y/%m/%d/%H/%M")

    # random part to avoid overwrites
    file_uuid = str(uuid.uuid4())[:8]

    return f"{base_path}{date_part}/merged_logs_{file_uuid}.json.gz"
