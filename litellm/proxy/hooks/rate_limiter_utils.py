"""
Shared utility functions for rate limiter hooks.
"""

from typing import Optional, Union

from litellm.types.router import ModelGroupInfo
from litellm.types.utils import PriorityReservationDict


def convert_priority_to_percent(
    value: Union[float, PriorityReservationDict], model_info: Optional[ModelGroupInfo]
) -> float:
    """
    Convert priority reservation value to percentage (0.0-1.0).

    Supports three formats:
    1. Plain float/int: 0.9 -> 0.9 (90%)
    2. Dict with percent: {"type": "percent", "value": 0.9} -> 0.9
    3. Dict with rpm: {"type": "rpm", "value": 900} -> 900/model_rpm
    4. Dict with tpm: {"type": "tpm", "value": 900000} -> 900000/model_tpm

    Args:
        value: Priority value as float or dict with type/value keys
        model_info: Model configuration containing rpm/tpm limits

    Returns:
        float: Percentage value between 0.0 and 1.0
    """
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, dict):
        val_type = value.get("type", "percent")
        val_num = value.get("value", 1.0)

        if val_type == "percent":
            return float(val_num)
        elif (
            val_type == "rpm"
            and model_info
            and model_info.rpm
            and model_info.rpm > 0
        ):
            return float(val_num) / model_info.rpm
        elif (
            val_type == "tpm"
            and model_info
            and model_info.tpm
            and model_info.tpm > 0
        ):
            return float(val_num) / model_info.tpm

        # Fallback: treat as percent
        return float(val_num)
